# GNN para Predição de Atraso de Voos Brasileiros

TCC do MBA em IA & Big Data (ICMC-USP). O projeto treina uma **Graph Neural Network heterogênea** para prever o atraso de chegada (em minutos) de voos comerciais brasileiros, modelando as relações entre voos, aeroportos, aeronaves e clima como um grafo no Neo4j.

---

## Resultados (PoC — 2.360 voos, 80 épocas)

| Modelo | Parâmetros | Teste MAE | Teste RMSE | Melhor Época |
|--------|-----------|-----------|------------|-------------|
| **GAT** ← recomendado | 1.475.585 | **25,92 min** | **34,50 min** | 36 |
| HGT | 181.345 | 28,42 min | 37,46 min | 78 |
| TGN | 394.817 | 77,55 min | 101,81 min | 80 |

> O GAT converge mais rápido e tem o menor MAE. O HGT usa 8× menos parâmetros com resultado apenas 10% pior — bom para ambientes com memória limitada.

---

## Fluxo de Dados

```
APIs Externas          Neo4j (grafo)          PyTorch Geometric           Saída
──────────────         ─────────────          ─────────────────────────   ──────
VRA/ANAC     ──┐       ┌─ Flight              ┌─ data/graph.pt (HeteroData)
OpenSky      ──┼──▶    ├─ Airport     ──▶     │                             ──▶  models/model.pt
Open-Meteo   ──┘       ├─ Aircraft            └─ train.py / train_          ──▶  results/predictions.csv
                       └─ Clima                    minibatch.py
```

**Fases do pipeline:**
1. **Ingestão** → `load_data.py` popula o Neo4j com voos, aeroportos e clima
2. **Enriquecimento** → `enrich_aircraft_age.py` + `create_next_rotation.py` adicionam features e arestas
3. **Build** → `build_graph.py` extrai o Neo4j e gera `graph.pt` (HeteroData do PyG)
4. **Treino** → `train.py` ou `train_minibatch.py` treinam a GNN e salvam `model.pt`
5. **Inferência** → `predict.py` prevê atraso para voos conhecidos ou futuros

---

## Quick Start

```bash
# 1. Construir grafo de teste (PoC — ~2.360 voos, rápido)
python build_graph.py --sample 2000 --output data/graph_poc.pt --stats

# 2. Treinar o modelo GAT
python train.py --model gat --graph data/graph_poc.pt --output models/model_gat.pt --epochs 80

# 3. Ver previsões para todos os voos do grafo
python predict.py --graph data/graph_poc.pt --model models/model_gat.pt --model-type gat

# 4. Dataset completo (demora mais, usa mais RAM)
python build_graph.py --filter-airports --output data/graph_major.pt --stats
python train.py --model gat --graph data/graph_major.pt --output models/model_major.pt --epochs 80
```

---

## Requisitos

```
Python 3.11 (Anaconda)
torch
torch-geometric
neo4j (driver Python)
pandas
numpy
scikit-learn
requests
```

Neo4j rodando localmente em `bolt://192.168.15.118:7687` (usuário `neo4j`, senha `tcc12345`).
Credenciais OpenSky em `credentials.json` (não commitar).

---

## Documentação Detalhada

| Documento | O que cobre |
|-----------|-------------|
| **[PIPELINE.md](PIPELINE.md)** | Guia completo de execução — cada fase, cada comando, cada parâmetro |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Schema do grafo, feature engineering, arquitetura dos modelos, decisões de design |
| **[EXPERIMENTO_GNN.md](EXPERIMENTO_GNN.md)** | Log do experimento comparativo GAT × HGT × TGN com resultados detalhados |
