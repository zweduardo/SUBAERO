# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TCC (final work) for MBA in IA & Big Data — heterogeneous GNN to predict Brazilian flight arrival delays. Stack: Python 3.11 (Anaconda, CPU-only), Neo4j (local), PyTorch Geometric.

## Pipeline Commands

The pipeline runs in sequential phases. No build system or test runner exists; all commands are direct Python invocations.

```bash
# Phase 1: Data ingestion to Neo4j
python load_data.py                          # Full initial load (airports, flights, weather)
python enrich_aircraft_age.py               # Enrich Aircraft nodes with generation_age_years
python enrich_aircraft_age.py --dry-run
python create_next_rotation.py              # Create NEXT_ROTATION edges
python create_next_rotation.py --dry-run
python create_next_rotation.py --since 2026-01-01

# Utilities for weather reload/relink
python reload_clima_hourly.py
python link_clima.py

# Phase 2: Build PyG graph from Neo4j
python build_graph.py --output data/graph.pt --stats
python build_graph.py --sample 2000 --output data/graph_poc.pt --stats  # small PoC graph

# Phase 3: Train GNN
python train.py --model gat --graph data/graph.pt --output models/model_gat.pt --epochs 80
python train.py --model hgt --graph data/graph.pt --output models/model_hgt.pt --epochs 80
python train.py --model tgn --graph data/graph.pt --output models/model_tgn.pt --epochs 80

# Incremental fine-tune on new data
python update_graph.py --since 2026-03-01 --graph data/graph.pt --output data/graph.pt
python train.py --model gat --graph data/graph.pt --finetune models/model_gat.pt --output models/model_gat.pt --epochs 20

# Phase 4: Inference
python predict.py --graph data/graph.pt --model models/model_gat.pt --model-type gat
python predict.py --graph data/graph.pt --model models/model_gat.pt --model-type gat --output results/predictions.csv
python predict.py --graph data/graph.pt --model models/model_gat.pt --model-type gat --flights "FLT001,FLT002"
python predict.py --graph data/graph.pt --model models/model_gat.pt --model-type gat --future-since 2026-03-20
```

## Architecture

### Data Flow

```
External APIs → load_data.py → Neo4j Graph DB → build_graph.py → data/graph.pt → train.py → models/model.pt → predict.py
```

### Neo4j Graph Model

4 node types, 7 edge types:

```
Flight  -[ORIGIN]->            Airport
Flight  -[DESTINATION]->       Airport
Flight  -[NEXT_LEG]->          Flight       (same route, consecutive days)
Flight  -[NEXT_ROTATION]->     Flight       (same equipment_icao, same day)
Flight  -[ASSIGNED_TO]->       Aircraft
Flight  -[HAS_ORIGIN_WEATHER]->Clima
Clima   -[OBSERVED_AT]->       Airport
```

### GNN Models (`train.py`)

| Key | Class | Architecture | Params | Test MAE |
|-----|-------|-------------|--------|---------|
| `gat` | `FlightDelayGAT` | 2-layer HeteroConv(GATConv) | 1.47M | 25.92 min |
| `hgt` | `FlightDelayHGT` | 2-layer HGTConv | 181K | 28.42 min |
| `tgn` | `FlightDelayTGN` | GRU memory + HeteroConv | 394K | 77.55 min |

**GAT is the best-performing model.** All models use `ToUndirected()` during training to add reverse edges.

### Node Features

| Node | Features |
|------|---------|
| Flight | `hour_dep`, `duration_min`, `delay_dep_min_clipped` (clipped [-30, 300]) |
| Airport | `latitude`, `longitude`, `elevation` |
| Aircraft | `generation_age_years` |
| Clima | `temp`, `windspeed`, `rain`, `clouds` |

**Target:** `delay_arrival_min` — z-score normalized, clipped to [-30, 360] min before normalization.

### Key Files

| File | Role |
|------|------|
| `api_calls.py` | All external API wrappers (ANAC VRA, OpenWeatherMap, Open-Meteo, OpenSky) with retry/backoff |
| `load_data.py` | Full initial data ingestion pipeline; contains the `Neo4jDB` class |
| `build_graph.py` | Neo4j → PyTorch Geometric `HeteroData`; handles feature engineering and z-score normalization |
| `train.py` | GNN model definitions and training loop; model registry via `--model` flag |
| `predict.py` | Inference; supports known flights, specific IDs, or future flights from Neo4j |
| `update_graph.py` | Incremental graph update; tags new nodes with `new_mask=True` for fine-tuning |
| `data/airports.csv` | OurAirports database (Brazilian airports, used by load_data.py) |
| `data/periods_split.json` | Period-based train/val/test splits with airport coverage |
| `docs/EXPERIMENTO_GNN.md` | Full experiment log with architecture decisions, hyperparameters, and results |

### Obsolete Files

`build_train_GAT.py` (v1, homogeneous) and `build_train_GAT_v2.py` (v2, HeteroGAT) are superseded prototypes. The current production pipeline is `build_graph.py` + `train.py`.

## Configuration

Neo4j connection is hardcoded in every script:
```python
NEO4J_URI      = "bolt://192.168.15.118:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "tcc12345"
```

OpenSky OAuth2 credentials are loaded from `credentials.json` at runtime. This file is sensitive and should not be committed.

## Dependencies

No `requirements.txt` exists. Required packages: `torch`, `torch_geometric`, `neo4j`, `pandas`, `numpy`, `scikit-learn`, `requests`.
