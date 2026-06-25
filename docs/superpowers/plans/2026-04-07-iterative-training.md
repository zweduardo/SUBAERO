# Iterative Per-Airport Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `build_train_iterative.py` — a memory-efficient training script that iterates over `(airport, month)` subgraphs lazily, with GAT and HGT model support.

**Architecture:** Periods `(iata_code, year, month)` are split 70/15/15 into train/val/test sets with airport coverage guarantees. Each epoch iterates through all train periods in shuffled order, building and discarding one subgraph at a time. HGT is ported from `train.py` and uses the full heterogeneous graph (flight + airport + aircraft + clima), while GAT uses only flight→flight edges.

**Tech Stack:** Python 3.11, PyTorch, PyTorch Geometric (`HeteroData`, `GATConv`, `HGTConv`), Neo4j (bolt), pandas, numpy, scikit-learn, argparse.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `build_train_iterative.py` | **Create** | Main script: Neo4j queries, period split, graph builder, models, training loop, output |
| `docs/superpowers/specs/2026-04-07-iterative-training-design.md` | Exists | Spec — do not modify |

All logic lives in a single file to match the existing prototype pattern in this repo.

---

## Task 1: File skeleton and CLI args

**Files:**
- Create: `build_train_iterative.py`

- [ ] **Step 1: Create the file with imports and config block**

```python
# build_train_iterative.py
# Iterative per-airport training — GAT and HGT on (airport, month) subgraphs.

import argparse
import json
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from neo4j import GraphDatabase
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HGTConv

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://192.168.15.118:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "tcc12345"

FLIGHT_FEAT_COLS   = ["hour_dep", "duration_min", "delay_dep_min_clipped"]
AIRPORT_FEAT_COLS  = ["latitude", "longitude", "elevation"]
AIRCRAFT_FEAT_COLS = ["generation_age_years"]
CLIMA_FEAT_COLS    = ["temp", "windspeed", "rain", "clouds"]
TARGET_COL         = "delay_arrival_min"
```

- [ ] **Step 2: Add CLI argument parser at the bottom of the file**

```python
# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Iterative per-airport GNN training")
    p.add_argument("--model",     default="gat",                choices=["gat", "hgt"])
    p.add_argument("--epochs",    default=50,   type=int)
    p.add_argument("--lr",        default=1e-3, type=float)
    p.add_argument("--wd",        default=1e-4, type=float)
    p.add_argument("--hidden",    default=64,   type=int)
    p.add_argument("--heads",     default=4,    type=int)
    p.add_argument("--dropout",   default=0.3,  type=float)
    p.add_argument("--seed",      default=42,   type=int)
    p.add_argument("--log-every", default=5,    type=int)
    p.add_argument("--output",    default="model_iterative.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
```

- [ ] **Step 3: Add stub `main(args)` function so the file is runnable**

```python
def main(args):
    print(f"[config] model={args.model} epochs={args.epochs} seed={args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
```

- [ ] **Step 4: Verify the file runs without errors**

```bash
python build_train_iterative.py --model gat --epochs 5
```

Expected output:
```
[config] model=gat epochs=5 seed=42
```

- [ ] **Step 5: Commit**

```bash
git add build_train_iterative.py
git commit -m "feat: add build_train_iterative skeleton with CLI args"
```

---

## Task 2: Neo4j query functions

**Files:**
- Modify: `build_train_iterative.py`

- [ ] **Step 1: Add `Neo4jLoader` class with the four required query methods**

Add this class after the config block:

```python
# ─────────────────────────────────────────────────────────────────────────────
#  Neo4j data access
# ─────────────────────────────────────────────────────────────────────────────
class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cypher, parameters=None):
        with self.driver.session() as session:
            return session.run(cypher, parameters).data()

    def get_large_airports(self):
        """Returns list of IATA codes for large_airport type nodes."""
        rows = self.query("""
            MATCH (a:Airport {type: 'large_airport'})
            WHERE a.iata_code IS NOT NULL
            RETURN a.iata_code AS iata_code
        """)
        return [r["iata_code"] for r in rows if r["iata_code"]]

    def get_available_periods(self, iata_codes):
        """Returns list of (iata_code, year, month) tuples with >=1 flight."""
        rows = self.query("""
            MATCH (f:Flight)-[:ORIGIN]->(a:Airport)
            WHERE a.iata_code IN $codes
              AND f.scheduled_departure IS NOT NULL
            RETURN a.iata_code AS iata_code,
                   toInteger(substring(toString(f.scheduled_departure), 0, 4)) AS year,
                   toInteger(substring(toString(f.scheduled_departure), 5, 2)) AS month,
                   count(f) AS n_flights
            ORDER BY iata_code, year, month
        """, parameters={"codes": iata_codes})
        return [(r["iata_code"], r["year"], r["month"]) for r in rows if r["n_flights"] > 0]

    def get_flight_data_by_period(self, iata_code, year, month):
        """Returns all flight rows touching iata_code in the given year/month."""
        rows = self.query("""
            MATCH (f:Flight)-[:ORIGIN]->(a1:Airport),
                  (f)-[:DESTINATION]->(a2:Airport),
                  (f)-[:ASSIGNED_TO]->(ac:Aircraft),
                  (f)-[:HAS_ORIGIN_WEATHER]->(c:Clima)
            WHERE (a1.iata_code = $iata OR a2.iata_code = $iata)
              AND toInteger(substring(toString(f.scheduled_departure), 0, 4)) = $year
              AND toInteger(substring(toString(f.scheduled_departure), 5, 2)) = $month
            RETURN f.flight_id          AS flight_id,
                   f.scheduled_departure AS scheduled_departure,
                   f.scheduled_arrival   AS scheduled_arrival,
                   f.delay_departure_min AS delay_departure_min,
                   f.delay_arrival_min   AS delay_arrival_min,
                   a1.iata_code AS dep_iata,
                   a1.latitude  AS dep_latitude,
                   a1.longitude AS dep_longitude,
                   a1.elevation AS dep_elevation,
                   a2.iata_code AS arr_iata,
                   a2.latitude  AS arr_latitude,
                   a2.longitude AS arr_longitude,
                   a2.elevation AS arr_elevation,
                   ac.equipment_icao        AS equipment_icao,
                   ac.generation_age_years  AS generation_age_years,
                   c.clima_id  AS clima_id,
                   c.temp      AS temp,
                   c.windspeed AS windspeed,
                   c.rain      AS rain,
                   c.clouds    AS clouds
        """, parameters={"iata": iata_code, "year": year, "month": month})
        return rows

    def get_next_leg_edges_for_flights(self, flight_ids):
        """Returns NEXT_LEG edges between flights in the given id set."""
        rows = self.query("""
            MATCH (f1:Flight)-[:NEXT_LEG]->(f2:Flight)
            WHERE f1.flight_id IN $ids AND f2.flight_id IN $ids
            RETURN f1.flight_id AS src_id, f2.flight_id AS dst_id
        """, parameters={"ids": flight_ids})
        return rows
```

- [ ] **Step 2: In `main(args)`, connect to Neo4j and print available periods**

```python
def main(args):
    print(f"[config] model={args.model} epochs={args.epochs} seed={args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        airports = loader.get_large_airports()
        print(f"[data] large airports found: {len(airports)} — {airports}")
        periods = loader.get_available_periods(airports)
        print(f"[data] available periods: {len(periods)}")
    finally:
        loader.close()
```

- [ ] **Step 3: Verify Neo4j queries return data**

```bash
python build_train_iterative.py --model gat --epochs 1
```

Expected output (numbers will vary):
```
[config] model=gat epochs=1 seed=42
[data] large airports found: 12 — ['GRU', 'VCP', 'GIG', ...]
[data] available periods: 47
```

If `large airports found: 0`, check the Neo4j `type` property value:
```cypher
MATCH (a:Airport) RETURN DISTINCT a.type LIMIT 20
```

- [ ] **Step 4: Commit**

```bash
git add build_train_iterative.py
git commit -m "feat: add Neo4j query functions for large airports and periods"
```

---

## Task 3: Period split with airport coverage guarantee

**Files:**
- Modify: `build_train_iterative.py`

- [ ] **Step 1: Add `split_periods()` function**

```python
# ─────────────────────────────────────────────────────────────────────────────
#  Period split
# ─────────────────────────────────────────────────────────────────────────────
def split_periods(periods, seed=42, train_ratio=0.70, val_ratio=0.15):
    """
    Split (iata, year, month) periods into train/val/test.

    Guarantee: if an airport has >= 3 periods, at least 1 appears in each
    split. If it has 2, assign 1 to train and 1 to val. If only 1, assign
    to train.

    Returns: (train_periods, val_periods, test_periods), each a list of tuples.
    """
    rng = random.Random(seed)

    # Group by airport
    from collections import defaultdict
    by_airport = defaultdict(list)
    for p in periods:
        by_airport[p[0]].append(p)

    train, val, test = [], [], []

    for iata, airport_periods in by_airport.items():
        shuffled = airport_periods[:]
        rng.shuffle(shuffled)
        n = len(shuffled)

        if n == 1:
            train.append(shuffled[0])
        elif n == 2:
            train.append(shuffled[0])
            val.append(shuffled[1])
        else:
            # Guarantee at least 1 in each split, distribute the rest proportionally
            test.append(shuffled[0])
            val.append(shuffled[1])
            train.append(shuffled[2])
            remaining = shuffled[3:]
            for p in remaining:
                r = rng.random()
                if r < train_ratio:
                    train.append(p)
                elif r < train_ratio + val_ratio:
                    val.append(p)
                else:
                    test.append(p)

    # Final shuffle within each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test
```

- [ ] **Step 2: Call `split_periods()` in `main()` and print the split summary**

Replace the periods print line in `main()`:

```python
        periods = loader.get_available_periods(airports)
        print(f"[data] available periods: {len(periods)}")

        train_periods, val_periods, test_periods = split_periods(periods, seed=args.seed)
        print(f"[split] train={len(train_periods)} val={len(val_periods)} test={len(test_periods)}")

        # Save split metadata for reproducibility
        split_meta = {
            "train": [list(p) for p in train_periods],
            "val":   [list(p) for p in val_periods],
            "test":  [list(p) for p in test_periods],
        }
        with open("periods_split.json", "w") as f:
            json.dump(split_meta, f, indent=2)
        print("[data] period split saved to periods_split.json")
```

- [ ] **Step 3: Verify split runs and each airport appears in train**

```bash
python build_train_iterative.py --model gat --epochs 1
```

Expected:
```
[split] train=33 val=7 test=7
[data] period split saved to periods_split.json
```

Then check the JSON:
```bash
python -c "import json; d=json.load(open('periods_split.json')); print('train airports:', sorted(set(p[0] for p in d['train'])))"
```

- [ ] **Step 4: Commit**

```bash
git add build_train_iterative.py
git commit -m "feat: add period split with per-airport coverage guarantee"
```

---

## Task 4: Feature engineering and graph builder

**Files:**
- Modify: `build_train_iterative.py`

This is ported directly from `build_train_GAT.py` with one change: instead of taking pre-loaded DataFrames, `build_period_graph()` accepts raw rows from Neo4j and builds the full `HeteroData` internally.

- [ ] **Step 1: Add feature engineering functions (copy from `build_train_GAT.py`)**

```python
# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def build_flight_features(df):
    df = df.copy()
    dep = pd.to_datetime(df["scheduled_departure"], errors="coerce", utc=True)
    arr = pd.to_datetime(df["scheduled_arrival"],   errors="coerce", utc=True)
    df["hour_dep"]            = dep.dt.hour.astype(float)
    df["duration_min"]        = (arr - dep).dt.total_seconds() / 60.0
    df["delay_dep_min_clipped"] = (
        pd.to_numeric(df["delay_departure_min"], errors="coerce")
        .fillna(0.0).clip(-30, 300)
    )
    return df


def build_airport_features(df):
    df = df.copy()
    for col in AIRPORT_FEAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def build_aircraft_features(df):
    df = df.copy()
    df["generation_age_years"] = pd.to_numeric(
        df["generation_age_years"], errors="coerce"
    )
    med = df["generation_age_years"].median()
    df["generation_age_years"] = df["generation_age_years"].fillna(
        med if not math.isnan(med) else 15.0
    )
    return df


def build_clima_features(df):
    df = df.copy()
    for col in CLIMA_FEAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def scale_features(df, cols):
    """Z-score normalize columns; returns (float32 array, fitted StandardScaler)."""
    X = df[cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    scaler = StandardScaler()
    return scaler.fit_transform(X).astype(np.float32), scaler
```

- [ ] **Step 2: Add `build_period_graph()` function**

```python
# ─────────────────────────────────────────────────────────────────────────────
#  Graph builder
# ─────────────────────────────────────────────────────────────────────────────
def build_period_graph(loader, iata_code, year, month):
    """
    Builds a HeteroData subgraph for one (airport, month) period.
    Returns (graph, y_mean, y_std) — caller must del graph after use.
    Returns None if the period has fewer than 10 flights.
    """
    rows = loader.get_flight_data_by_period(iata_code, year, month)
    if not rows:
        return None, None, None

    flight_df = pd.DataFrame(rows)
    if len(flight_df) < 10:
        return None, None, None

    # ── Feature engineering ─────────────────────────────────────────────────
    flight_df = build_flight_features(flight_df)

    # Deduplicate node tables from the denormalized flight rows
    airport_df = pd.concat([
        flight_df[["dep_iata", "dep_latitude", "dep_longitude", "dep_elevation"]]
            .rename(columns={"dep_iata": "iata_code", "dep_latitude": "latitude",
                             "dep_longitude": "longitude", "dep_elevation": "elevation"}),
        flight_df[["arr_iata", "arr_latitude", "arr_longitude", "arr_elevation"]]
            .rename(columns={"arr_iata": "iata_code", "arr_latitude": "latitude",
                             "arr_longitude": "longitude", "arr_elevation": "elevation"}),
    ]).drop_duplicates("iata_code").reset_index(drop=True)

    aircraft_df = flight_df[["equipment_icao", "generation_age_years"]].drop_duplicates(
        "equipment_icao"
    ).reset_index(drop=True)

    clima_df = flight_df[["clima_id", "temp", "windspeed", "rain", "clouds"]].drop_duplicates(
        "clima_id"
    ).reset_index(drop=True)

    airport_df  = build_airport_features(airport_df)
    aircraft_df = build_aircraft_features(aircraft_df)
    clima_df    = build_clima_features(clima_df)

    # ── Normalize features ──────────────────────────────────────────────────
    flight_feats,   _ = scale_features(flight_df,   FLIGHT_FEAT_COLS)
    airport_feats,  _ = scale_features(airport_df,  AIRPORT_FEAT_COLS)
    aircraft_feats, _ = scale_features(aircraft_df, AIRCRAFT_FEAT_COLS)
    clima_feats,    _ = scale_features(clima_df,     CLIMA_FEAT_COLS)

    # ── Target ──────────────────────────────────────────────────────────────
    y_raw = (
        pd.to_numeric(flight_df[TARGET_COL], errors="coerce")
        .fillna(0.0).clip(-30, 360).values.astype(np.float32)
    )
    y_mean, y_std = float(y_raw.mean()), float(y_raw.std()) + 1e-8
    y_norm = (y_raw - y_mean) / y_std

    # ── Build HeteroData ────────────────────────────────────────────────────
    graph = HeteroData()
    graph["flight"].x   = torch.tensor(flight_feats,   dtype=torch.float32)
    graph["airport"].x  = torch.tensor(airport_feats,  dtype=torch.float32)
    graph["aircraft"].x = torch.tensor(aircraft_feats, dtype=torch.float32)
    graph["clima"].x    = torch.tensor(clima_feats,    dtype=torch.float32)
    graph["flight"].y   = torch.tensor(y_norm,         dtype=torch.float32)
    graph["flight"].y_mean = y_mean
    graph["flight"].y_std  = y_std

    # ── Index maps ──────────────────────────────────────────────────────────
    airport_idx  = {v: i for i, v in enumerate(airport_df["iata_code"])       if pd.notna(v)}
    aircraft_idx = {v: i for i, v in enumerate(aircraft_df["equipment_icao"]) if pd.notna(v)}
    clima_idx    = {v: i for i, v in enumerate(clima_df["clima_id"])           if pd.notna(v)}
    flight_idx   = {v: i for i, v in enumerate(flight_df["flight_id"])         if pd.notna(v)}
    n_flights    = len(flight_df)
    frange       = np.arange(n_flights)

    dep_idx = flight_df["dep_iata"].map(airport_idx).fillna(-1).astype(int).values
    arr_idx = flight_df["arr_iata"].map(airport_idx).fillna(-1).astype(int).values
    ac_idx  = flight_df["equipment_icao"].map(aircraft_idx).fillna(-1).astype(int).values
    cl_idx  = flight_df["clima_id"].map(clima_idx).fillna(-1).astype(int).values

    def _edge(src, dst):
        mask = (src >= 0) & (dst >= 0)
        return torch.tensor(np.array([src[mask], dst[mask]]), dtype=torch.long)

    graph["flight", "departs_from", "airport"].edge_index  = _edge(frange, dep_idx)
    graph["flight", "arrives_at",   "airport"].edge_index  = _edge(frange, arr_idx)
    graph["flight", "operated_by",  "aircraft"].edge_index = _edge(frange, ac_idx)
    graph["flight", "has_clima",    "clima"].edge_index    = _edge(frange, cl_idx)

    # NEXT_LEG edges (within this period only)
    flight_ids = flight_df["flight_id"].dropna().tolist()
    nl_rows = loader.get_next_leg_edges_for_flights(flight_ids)
    if nl_rows:
        nl_df = pd.DataFrame(nl_rows)
        nl_src = nl_df["src_id"].map(flight_idx).dropna().astype(int)
        nl_dst = nl_df["dst_id"].map(flight_idx).dropna().astype(int)
        valid = nl_src.index.intersection(nl_dst.index)
        graph["flight", "next_leg", "flight"].edge_index = torch.tensor(
            np.array([nl_src.loc[valid].values, nl_dst.loc[valid].values]),
            dtype=torch.long,
        )
    else:
        idx = np.arange(n_flights)
        graph["flight", "next_leg", "flight"].edge_index = torch.tensor(
            np.array([idx, idx]), dtype=torch.long
        )

    return graph, y_mean, y_std
```

- [ ] **Step 3: Test graph builder in `main()` by building the first train period**

Add to `main()` after the split:

```python
        # Smoke-test graph builder on first train period
        iata, yr, mo = train_periods[0]
        print(f"[test] building graph for {iata} {yr}-{mo:02d} ...")
        graph, y_mean, y_std = build_period_graph(loader, iata, yr, mo)
        if graph is not None:
            print(f"[test] flights={graph['flight'].x.shape[0]}  y_mean={y_mean:.1f}  y_std={y_std:.1f}")
            del graph
        else:
            print("[test] period returned None (too few flights)")
```

- [ ] **Step 4: Run smoke test**

```bash
python build_train_iterative.py --model gat --epochs 1
```

Expected (numbers vary):
```
[test] building graph for GRU 2025-01 ...
[test] flights=842  y_mean=8.3  y_std=25.1
```

- [ ] **Step 5: Commit**

```bash
git add build_train_iterative.py
git commit -m "feat: add feature engineering and per-period graph builder"
```

---

## Task 5: GAT and HGT model definitions

**Files:**
- Modify: `build_train_iterative.py`

- [ ] **Step 1: Add `FlightGAT` model (homogeneous, flight→flight only)**

```python
# ─────────────────────────────────────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────────────────────────────────────
class FlightGAT(nn.Module):
    """Homogeneous GAT on flight nodes connected via next_leg edges."""

    def __init__(self, in_channels, hidden=64, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden, heads=heads, concat=True,
                             add_self_loops=False)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, concat=False,
                             add_self_loops=False)
        self.head  = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        x  = x_dict["flight"]
        ei = edge_index_dict[("flight", "next_leg", "flight")]
        x  = F.elu(self.conv1(x, ei))
        x  = F.dropout(x, p=self.dropout, training=self.training)
        x  = self.conv2(x, ei)
        return self.head(x).squeeze(-1)
```

- [ ] **Step 2: Add `FlightHGT` model (heterogeneous, all node types)**

```python
class FlightHGT(nn.Module):
    """Heterogeneous Graph Transformer using all node types."""

    def __init__(self, in_channels_dict, hidden=64, heads=4, dropout=0.3,
                 metadata=None):
        super().__init__()
        self.dropout = dropout

        # Project each node type to shared hidden dim
        self.proj = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden)
            for ntype, in_dim in in_channels_dict.items()
        })

        self.conv1 = HGTConv(hidden, hidden, metadata, heads=heads)
        self.conv2 = HGTConv(hidden, hidden, metadata, heads=heads)

        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        # Project all node types to hidden dim
        h = {ntype: F.elu(self.proj[ntype](x)) for ntype, x in x_dict.items()}

        h = self.conv1(h, edge_index_dict)
        h = {k: F.elu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)

        return self.head(h["flight"]).squeeze(-1)
```

- [ ] **Step 3: Add `build_model()` factory function**

```python
def build_model(args, graph):
    """Instantiate the requested model from a sample graph."""
    in_channels_dict = {ntype: graph[ntype].x.shape[1] for ntype in graph.node_types}
    metadata = (graph.node_types, graph.edge_types)

    if args.model == "gat":
        model = FlightGAT(
            in_channels=in_channels_dict["flight"],
            hidden=args.hidden,
            heads=args.heads,
            dropout=args.dropout,
        )
    else:  # hgt
        model = FlightHGT(
            in_channels_dict=in_channels_dict,
            hidden=args.hidden,
            heads=args.heads,
            dropout=args.dropout,
            metadata=metadata,
        )
    return model
```

- [ ] **Step 4: Verify model builds and does a forward pass**

Add to `main()` smoke-test block (after the graph smoke-test):

```python
        # Build model and do a dummy forward
        if graph is not None:
            graph, y_mean, y_std = build_period_graph(loader, iata, yr, mo)
            model = build_model(args, graph)
            with torch.no_grad():
                out = model(graph.x_dict, graph.edge_index_dict)
            print(f"[test] model={args.model}  output shape={out.shape}  params={sum(p.numel() for p in model.parameters()):,}")
            del graph
```

```bash
python build_train_iterative.py --model gat --epochs 1
python build_train_iterative.py --model hgt --epochs 1
```

Expected (shapes and param counts will vary):
```
[test] model=gat  output shape=torch.Size([842])  params=84,481
[test] model=hgt  output shape=torch.Size([842])  params=181,345
```

- [ ] **Step 5: Commit**

```bash
git add build_train_iterative.py
git commit -m "feat: add FlightGAT and FlightHGT model definitions"
```

---

## Task 6: Training loop

**Files:**
- Modify: `build_train_iterative.py`

- [ ] **Step 1: Add `train_step()` and `evaluate()` helper functions**

```python
# ─────────────────────────────────────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def train_step(model, graph, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
    y = graph["flight"].y.to(device)
    out = model(x_dict, ei_dict)
    loss = criterion(out, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def evaluate(model, graph, criterion, device):
    """Returns (mse_loss, mae_minutes)."""
    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
        ei_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
        y_norm = graph["flight"].y.to(device)
        out = model(x_dict, ei_dict)
        loss = criterion(out, y_norm).item()

        y_mean = graph["flight"].y_mean
        y_std  = graph["flight"].y_std
        pred_min = out.cpu() * y_std + y_mean
        true_min = y_norm.cpu() * y_std + y_mean
        mae = float((pred_min - true_min).abs().mean())
    return loss, mae
```

- [ ] **Step 2: Replace `main()` with the full training loop**

Remove the smoke-test block and replace the body of `main()` after the split with:

```python
    loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        airports = loader.get_large_airports()
        print(f"[data] large airports: {len(airports)}")
        periods = loader.get_available_periods(airports)
        print(f"[data] available periods: {len(periods)}")

        train_periods, val_periods, test_periods = split_periods(periods, seed=args.seed)
        print(f"[split] train={len(train_periods)} val={len(val_periods)} test={len(test_periods)}")

        with open("periods_split.json", "w") as f:
            json.dump({
                "train": [list(p) for p in train_periods],
                "val":   [list(p) for p in val_periods],
                "test":  [list(p) for p in test_periods],
            }, f, indent=2)

        # ── Build model from first valid train period ──────────────────────
        model = None
        for period in train_periods:
            sample_graph, _, _ = build_period_graph(loader, *period)
            if sample_graph is not None:
                model = build_model(args, sample_graph)
                del sample_graph
                break
        if model is None:
            raise RuntimeError("No valid period found to initialize model.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"[model] {args.model.upper()} — {sum(p.numel() for p in model.parameters()):,} params — device={device}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        criterion = nn.MSELoss()

        best_val_mae = float("inf")
        best_state   = None
        metrics      = []  # (epoch, train_mse_avg, val_mae_avg)

        # ── Epoch loop ─────────────────────────────────────────────────────
        for epoch in range(1, args.epochs + 1):
            # Train
            epoch_train_losses = []
            epoch_order = train_periods[:]
            random.shuffle(epoch_order)

            for period in epoch_order:
                graph, _, _ = build_period_graph(loader, *period)
                if graph is None:
                    continue
                loss = train_step(model, graph, optimizer, criterion, device)
                epoch_train_losses.append(loss)
                del graph

            avg_train = float(np.mean(epoch_train_losses)) if epoch_train_losses else 0.0

            # Validate
            val_maes = []
            for period in val_periods:
                graph, _, _ = build_period_graph(loader, *period)
                if graph is None:
                    continue
                _, mae = evaluate(model, graph, criterion, device)
                val_maes.append(mae)
                del graph

            avg_val_mae = float(np.mean(val_maes)) if val_maes else 0.0
            scheduler.step(avg_val_mae)

            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            metrics.append((epoch, avg_train, avg_val_mae))

            if epoch % args.log_every == 0:
                print(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train MSE: {avg_train:.4f} | "
                    f"Val MAE: {avg_val_mae:.1f} min | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

        # ── Test ───────────────────────────────────────────────────────────
        model.load_state_dict(best_state)
        test_maes = []
        for period in test_periods:
            graph, _, _ = build_period_graph(loader, *period)
            if graph is None:
                continue
            _, mae = evaluate(model, graph, criterion, device)
            test_maes.append(mae)
            del graph

        avg_test_mae = float(np.mean(test_maes)) if test_maes else 0.0
        print(f"\nTest MAE (best model): {avg_test_mae:.1f} min")

        # ── Save outputs ───────────────────────────────────────────────────
        torch.save(best_state, args.output)
        print(f"Model saved to '{args.output}'")

        import csv
        with open("metrics_iterative.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_mse", "val_mae"])
            writer.writerows(metrics)
        print("Metrics saved to 'metrics_iterative.csv'")

    finally:
        loader.close()
```

Note: `args.log_every` — argparse stores `--log-every` as `log_every` (hyphens become underscores).

- [ ] **Step 3: Run a short end-to-end smoke test (2 epochs)**

```bash
python build_train_iterative.py --model gat --epochs 2 --log-every 1
```

Expected output:
```
[data] large airports: 12
[data] available periods: 47
[split] train=33 val=7 test=7
[model] GAT — 84,481 params — device=cpu
Epoch   1/2 | Train MSE: 0.9134 | Val MAE: 31.2 min | LR: 0.001000
Epoch   2/2 | Train MSE: 0.8801 | Val MAE: 29.7 min | LR: 0.001000

Test MAE (best model): 28.4 min
Model saved to 'model_iterative.pt'
Metrics saved to 'metrics_iterative.csv'
```

- [ ] **Step 4: Test HGT as well**

```bash
python build_train_iterative.py --model hgt --epochs 2 --log-every 1
```

Expected: same format, different MAE numbers.

- [ ] **Step 5: Commit**

```bash
git add build_train_iterative.py
git commit -m "feat: add iterative training loop with lazy per-period graph loading"
```

---

## Task 7: Final cleanup and output verification

**Files:**
- Modify: `build_train_iterative.py`

- [ ] **Step 1: Remove smoke-test code from `main()` if any remains**

Check that `main()` no longer contains any `[test]` print lines or temporary graph-building code that was added for verification during earlier tasks.

- [ ] **Step 2: Run a full training session (50 epochs) and verify all output files exist**

```bash
python build_train_iterative.py --model gat --epochs 50 --log-every 10
```

After completion verify:

```bash
python -c "
import torch, json, os
assert os.path.exists('model_iterative.pt'), 'model missing'
assert os.path.exists('periods_split.json'), 'split missing'
assert os.path.exists('metrics_iterative.csv'), 'metrics missing'
state = torch.load('model_iterative.pt', map_location='cpu')
print('model keys:', list(state.keys())[:3], '...')
split = json.load(open('periods_split.json'))
print('split sizes — train:', len(split['train']), 'val:', len(split['val']), 'test:', len(split['test']))
import csv
rows = list(csv.reader(open('metrics_iterative.csv')))
print('metrics rows:', len(rows)-1, '(header + epochs)')
"
```

Expected:
```
model keys: ['conv1.lin_src.weight', 'conv1.lin_dst.weight', 'conv1.att_src'] ...
split sizes — train: 33 val: 7 test: 7
metrics rows: 50 (header + epochs)
```

- [ ] **Step 3: Final commit**

```bash
git add build_train_iterative.py
git commit -m "feat: complete build_train_iterative.py with GAT/HGT and lazy period loading"
```
