# Design: Iterative Per-Airport Training (build_train_iterative.py)

**Date:** 2026-04-07  
**Status:** Approved

## Context

The existing `build_train_GAT.py` is a homogeneous prototype that loads the entire flight graph into RAM at once. The goal here is to create `build_train_iterative.py` — a new experimental script that:

1. Trains on subgraphs split by `(airport, month)` periods instead of the full graph
2. Keeps only one subgraph in RAM at a time (lazy loading per period)
3. Adds HGT as a second model option alongside the existing GAT
4. Enables curriculum-style training across Brazil's large airports for TCC experimentation

## Architecture

### Data Loading

**`get_large_airports() → list[str]`**  
Query Neo4j for airports with `type='large_airport'`. Returns list of IATA codes.

**`get_available_periods(airports) → list[tuple[str, int, int]]`**  
Query Neo4j for all `(iata_code, year, month)` combinations that have at least one flight departing or arriving at that airport. Returns list of period tuples.

**`build_period_graph(airport, year, month) → HeteroData`**  
Builds a heterogeneous PyG subgraph for the given period. Reuses the existing graph construction logic from `build_train_GAT.py` (node feature engineering, z-score normalization, edge index building). Caller is responsible for deleting the object after use to free RAM.

### Period Split

- Shuffle all available periods randomly (fixed seed for reproducibility)
- Split 70% train / 15% val / 15% test
- **Constraint:** If an airport has ≥ 3 periods, ensure at least 1 period appears in each split. If an airport has < 3 periods, assign to train first, then val.
- Split metadata saved to `periods_split.json` alongside the model output for reproducibility

### Models

**GAT (existing)**  
- 2-layer homogeneous GATConv on `(flight, next_leg, flight)` edges only
- Input: flight node features (3 dims)
- Hidden: 64, heads: 4

**HGT (new — ported from `train.py`)**  
- Linear projections per node type → shared hidden dim (64)
- 2x HGTConv (in=64, out=64, heads=4)
- Head: Linear(64→32) → ReLU → Dropout → Linear(32→1)
- Uses full heterogeneous graph (flight, airport, aircraft, clima)

Selected via `--model gat|hgt`.

### Training Loop

```
for epoch in range(epochs):
    shuffle(train_periods)
    for (airport, year, month) in train_periods:
        graph = build_period_graph(airport, year, month)
        train_step(model, graph)
        del graph                          # free RAM immediately

    val_mae = 0
    for (airport, year, month) in val_periods:
        graph = build_period_graph(airport, year, month)
        val_mae += evaluate(model, graph)
        del graph
    val_mae /= len(val_periods)

    log(epoch, val_mae)
    scheduler.step(val_mae)
    save_best_checkpoint()

# Final test
for (airport, year, month) in test_periods:
    graph = build_period_graph(airport, year, month)
    test_mae += evaluate(model, graph)
    del graph
```

- Optimizer: Adam (lr=1e-3, wd=1e-4)
- Scheduler: ReduceLROnPlateau (patience=10)
- Early stopping: based on val MAE
- Gradient clipping: max_norm=1.0
- Target: z-score normalized `delay_arrival_min`, clipped to [-30, 360] min

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gat` | Model architecture: `gat` or `hgt` |
| `--epochs` | `50` | Total training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--wd` | `1e-4` | Weight decay |
| `--hidden` | `64` | Hidden dimension |
| `--heads` | `4` | Attention heads |
| `--dropout` | `0.3` | Dropout probability |
| `--seed` | `42` | Random seed for period shuffle |
| `--log-every` | `5` | Log frequency (epochs) |
| `--output` | `model_iterative.pt` | Output model path |

### Output Files

- `model_iterative.pt` — best model state dict
- `periods_split.json` — train/val/test period assignment
- `metrics_iterative.csv` — epoch-level metrics (epoch, train_loss, val_mae)

## Verification

1. Run `python build_train_iterative.py --model gat --epochs 5` — should complete without OOM, log MAE per epoch
2. Run `python build_train_iterative.py --model hgt --epochs 5` — HGT should train and produce lower/comparable MAE to GAT
3. Check `periods_split.json` — each large airport should appear in at least train + one of val/test
4. Check `metrics_iterative.csv` — val MAE should generally decrease over epochs
