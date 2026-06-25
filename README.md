# SUBAERO — Predicting Flight Delay Propagation with Heterogeneous GNNs

This project trains a **Heterogeneous Graph Neural Network** to predict arrival delays (in minutes) of Brazilian commercial flights, modeling the relationships between flights, airports, aircraft, and weather as a graph in Neo4j.

---

## Main Results (National Dataset: ~1.17M flights)

Using the **Heterogeneous Graph Transformer (HGT)** architecture with a *subgraph partitioning* strategy to handle memory limitations, the model achieved a **Mean Absolute Error (MAE) of 6.7 minutes** on the validation set.

To validate generalization capability and avoid *overfitting*, an *out-of-sample* (holdout) validation completely excluded 5 airports (one from each region) from the training set. The MAE results for these airports, never seen by the model during training, were:

- **GYN (Goiânia):** 5.9 minutes
- **MAO (Manaus):** 6.5 minutes
- **FOR (Fortaleza):** 6.8 minutes
- **VCP (Campinas):** 6.8 minutes
- **POA (Porto Alegre):** 7.6 minutes

> The aggregate MAE for airports explicitly seen during training was very similar (6.9 minutes), demonstrating an excellent generalization capability of the model based on network topology and meteorological attributes. For reference, the GAT and TGN architectures achieved MAEs of 24.76 and 70.02 minutes in this same validated experiment, confirming the superiority of HGT.

---

## Data Flow

```
External APIs          Neo4j (graph)          PyTorch Geometric           Output
──────────────         ─────────────          ─────────────────────────   ──────
VRA/ANAC     ──┐       ┌─ Flight              ┌─ data/graph.pt (HeteroData)
OpenSky      ──┼──▶    ├─ Airport     ──▶     │                             ──▶  models/model.pt
Open-Meteo   ──┘       ├─ Aircraft            └─ train.py / train_          ──▶  results/predictions.csv
                       └─ Clima                    minibatch.py
```

**Pipeline phases:**
1. **Ingestion** → `load_data.py` populates Neo4j with flights, airports, and weather.
2. **Enrichment** → `enrich_aircraft_age.py` + `create_next_rotation.py` add features and edges.
3. **Build** → `build_graph.py` extracts from Neo4j and generates `graph.pt` (PyG HeteroData).
4. **Train** → `train.py` or `train_minibatch.py` train the GNN and save `model.pt`.
5. **Inference** → `predict.py` predicts delays for known or future flights.

---

## Quick Start

```bash
# 1. Build test graph (PoC — ~2,360 flights, fast)
python build_graph.py --sample 2000 --output data/graph_poc.pt --stats

# 2. Train the GAT model
python train.py --model gat --graph data/graph_poc.pt --output models/model_gat.pt --epochs 80

# 3. Predict for all flights in the graph
python predict.py --graph data/graph_poc.pt --model models/model_gat.pt --model-type gat

# 4. Full dataset (takes longer, uses more RAM)
python build_graph.py --filter-airports --output data/graph_major.pt --stats
python train.py --model gat --graph data/graph_major.pt --output models/model_major.pt --epochs 80
```

---

## Requirements

```
Python 3.11 (Anaconda)
torch
torch-geometric
neo4j (Python driver)
pandas
numpy
scikit-learn
requests
```

Neo4j running locally at `bolt://localhost:7687` (default user `neo4j`). Configure your credentials via environment variables.
OpenSky and OpenWeatherMap credentials should be configured via environment variables.

---

## Detailed Documentation

| Document | What it covers |
|-----------|-------------|
| **[PIPELINE.md](docs/PIPELINE.md)** | Complete execution guide — every phase, command, and parameter |
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Graph schema, feature engineering, model architecture, and design decisions |
| **[EXPERIMENTO_GNN.md](docs/EXPERIMENTO_GNN.md)** | Comparative experiment log (GAT × HGT × TGN) with detailed results |
