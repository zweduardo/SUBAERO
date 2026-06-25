# build_train_iterative_rotation.py
# Iterative per-airport training — GAT and HGT on (airport, month) subgraphs.
# Extends build_train_iterative.py by adding NEXT_ROTATION edges (same aircraft, same day).

import argparse
import json
import math
import random
import time
from datetime import datetime

import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from neo4j import GraphDatabase
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HGTConv, HeteroConv


#  Config
NEO4J_URI      = "bolt://192.168.15.118:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "tcc12345"

FLIGHT_FEAT_COLS   = ["hour_dep", "duration_min", "day_of_week", "delay_dep_min_clipped"]
AIRPORT_FEAT_COLS  = ["latitude", "longitude", "elevation"]
AIRCRAFT_FEAT_COLS = ["generation_age_years"]
CLIMA_FEAT_COLS    = ["temp", "windspeed", "rain", "clouds"]
DEST_CLIMA_COLS    = ["dest_temp", "dest_windspeed", "dest_rain", "dest_clouds"]
TARGET_COL         = "delay_arrival_min"



#  Neo4j data access
class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cypher, parameters=None):
        with self.driver.session() as session:
            return session.run(cypher, parameters).data()

    def get_large_airports(self):
        """Returns list of IATA codes for Brazilian large_airport and medium_airport nodes."""
        rows = self.query("""
            MATCH (a:Airport)
            WHERE a.iso_country = 'BR'
              AND a.type IN ['large_airport', 'medium_airport']
              AND a.iata_code IS NOT NULL
            RETURN DISTINCT a.iata_code AS iata_code
        """)
        return [r["iata_code"] for r in rows if r["iata_code"] and isinstance(r["iata_code"], str)]

    def get_available_periods(self, iata_codes):
        """Returns list of (iata_code, year, month) tuples with >=1 flight."""
        rows = self.query("""
            MATCH (f:Flight)-[:ORIGIN]->(a:Airport)
            WHERE a.iata_code IN $codes
              AND f.scheduled_departure IS NOT NULL
            RETURN a.iata_code AS iata_code,
                   toInteger(substring(f.scheduled_departure, 0, 4)) AS year,
                   toInteger(substring(f.scheduled_departure, 5, 2)) AS month,
                   count(f) AS n_flights
            ORDER BY iata_code, year, month
        """, parameters={"codes": iata_codes})
        return [(r["iata_code"], r["year"], r["month"]) for r in rows if r["n_flights"] > 0]

    def get_flight_data_by_period(self, iata_code, year, month):
        """Returns all flight rows touching iata_code in the given year/month.
        Uses string range on scheduled_departure to aproveitar o índice RANGE existente.
        """
        end_year  = year + 1 if month == 12 else year
        end_month = 1 if month == 12 else month + 1
        start = f"{year:04d}-{month:02d}-01T00:00:00"
        end   = f"{end_year:04d}-{end_month:02d}-01T00:00:00"
        rows = self.query("""
            MATCH (f:Flight)-[:ORIGIN]->(a1:Airport),
                  (f)-[:DESTINATION]->(a2:Airport),
                  (f)-[:ASSIGNED_TO]->(ac:Aircraft),
                  (f)-[:HAS_ORIGIN_WEATHER]->(c:Clima)
            WHERE (a1.iata_code = $iata OR a2.iata_code = $iata)
              AND f.scheduled_departure >= $start
              AND f.scheduled_departure < $end
            CALL (a2) {
                OPTIONAL MATCH (c_dest:Clima)-[:OBSERVED_AT]->(a2)
                RETURN c_dest LIMIT 1
            }
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
                   c.clouds    AS clouds,
                   c_dest.temp      AS dest_temp,
                   c_dest.windspeed AS dest_windspeed,
                   c_dest.rain      AS dest_rain,
                   c_dest.clouds    AS dest_clouds
        """, parameters={"iata": iata_code, "start": start, "end": end})
        return rows

    def get_next_leg_edges_for_flights(self, flight_ids):
        """Returns NEXT_LEG edges between flights in the given id set."""
        rows = self.query("""
            MATCH (f1:Flight)-[:NEXT_LEG]->(f2:Flight)
            WHERE f1.flight_id IN $ids AND f2.flight_id IN $ids
            RETURN f1.flight_id AS src_id, f2.flight_id AS dst_id
        """, parameters={"ids": flight_ids})
        return rows

    def get_next_rotation_edges_for_flights(self, flight_ids):
        """Returns NEXT_ROTATION edges (same aircraft, same day) between flights in the given id set."""
        rows = self.query("""
            MATCH (f1:Flight)-[:NEXT_ROTATION]->(f2:Flight)
            WHERE f1.flight_id IN $ids AND f2.flight_id IN $ids
            RETURN f1.flight_id AS src_id, f2.flight_id AS dst_id
        """, parameters={"ids": flight_ids})
        return rows


#  Period split
def split_periods(periods, seed=42, train_ratio=0.70, val_ratio=0.15):
    """
    Split (iata, year, month) periods into train/val/test.

    Guarantee: if an airport has >= 3 periods, at least 1 appears in each
    split. If it has 2, assign 1 to train and 1 to val. If only 1, assign
    to train.

    Returns: (train_periods, val_periods, test_periods), each a list of tuples.
    """
    from collections import defaultdict
    rng = random.Random(seed)

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
            # Guarantee at least 1 in each split
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

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


#  Feature engineering

def build_flight_features(df):
    df = df.copy()
    dep = pd.to_datetime(df["scheduled_departure"], errors="coerce", utc=True, format="mixed")
    arr = pd.to_datetime(df["scheduled_arrival"],   errors="coerce", utc=True, format="mixed")
    df["hour_dep"]              = dep.dt.hour.astype(float)
    df["duration_min"]          = (arr - dep).dt.total_seconds() / 60.0
    df["day_of_week"]           = dep.dt.dayofweek.astype(float)  # 0=Mon, 6=Sun
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
        med if not (isinstance(med, float) and math.isnan(med)) else 15.0
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



#  Graph builder

def build_period_graph(loader, iata_code, year, month):
    """
    Builds a HeteroData subgraph for one (airport, month) period.
    Returns (graph, y_mean, y_std) — caller must del graph after use.
    Returns (None, None, None) if the period has fewer than 10 flights.
    """
    rows = loader.get_flight_data_by_period(iata_code, year, month)
    if not rows:
        return None, None, None

    flight_df = pd.DataFrame(rows)
    if len(flight_df) < 10:
        return None, None, None

    # Feature engineering
    flight_df = build_flight_features(flight_df)

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

    for col in DEST_CLIMA_COLS:
        flight_df[col] = pd.to_numeric(flight_df.get(col, 0.0), errors="coerce").fillna(0.0)

    all_flight_cols = FLIGHT_FEAT_COLS + DEST_CLIMA_COLS
    flight_feats,   _ = scale_features(flight_df,   all_flight_cols)
    airport_feats,  _ = scale_features(airport_df,  AIRPORT_FEAT_COLS)
    aircraft_feats, _ = scale_features(aircraft_df, AIRCRAFT_FEAT_COLS)
    clima_feats,    _ = scale_features(clima_df,     CLIMA_FEAT_COLS)

    y_raw = (
        pd.to_numeric(flight_df[TARGET_COL], errors="coerce")
        .fillna(0.0).clip(-30, 360).values.astype(np.float32)
    )
    y_mean, y_std = float(y_raw.mean()), float(y_raw.std()) + 1e-8
    y_norm = (y_raw - y_mean) / y_std

    graph = HeteroData()
    graph["flight"].x   = torch.tensor(flight_feats,   dtype=torch.float32)
    graph["airport"].x  = torch.tensor(airport_feats,  dtype=torch.float32)
    graph["aircraft"].x = torch.tensor(aircraft_feats, dtype=torch.float32)
    graph["clima"].x    = torch.tensor(clima_feats,    dtype=torch.float32)
    graph["flight"].y      = torch.tensor(y_norm,  dtype=torch.float32)
    graph["flight"].y_mean = y_mean
    graph["flight"].y_std  = y_std

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

    flight_ids = flight_df["flight_id"].dropna().tolist()

    def _flight_edges(rows, rel_name):
        """Build edge_index for a flight->flight relation; falls back to self-loops."""
        if rows:
            df  = pd.DataFrame(rows)
            src = df["src_id"].map(flight_idx).dropna().astype(int)
            dst = df["dst_id"].map(flight_idx).dropna().astype(int)
            valid = src.index.intersection(dst.index)
            if len(valid) > 0:
                return torch.tensor(
                    np.array([src.loc[valid].values, dst.loc[valid].values]),
                    dtype=torch.long,
                )
        idx = np.arange(n_flights)
        return torch.tensor(np.array([idx, idx]), dtype=torch.long)

    nl_rows = loader.get_next_leg_edges_for_flights(flight_ids)
    graph["flight", "next_leg", "flight"].edge_index = _flight_edges(nl_rows, "next_leg")

    nr_rows = loader.get_next_rotation_edges_for_flights(flight_ids)
    graph["flight", "next_rotation", "flight"].edge_index = _flight_edges(nr_rows, "next_rotation")

    return graph, y_mean, y_std



#  Models

class FlightGAT(nn.Module):
    """
    GAT on flight nodes using both next_leg and next_rotation edges via HeteroConv.
    Messages from each relation are summed before the next layer.
    """

    def __init__(self, in_channels, hidden=64, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        self.conv1 = HeteroConv({
            ("flight", "next_leg",      "flight"): GATConv(in_channels, hidden, heads=heads, concat=True,  add_self_loops=False),
            ("flight", "next_rotation", "flight"): GATConv(in_channels, hidden, heads=heads, concat=True,  add_self_loops=False),
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("flight", "next_leg",      "flight"): GATConv(hidden * heads, hidden, heads=heads, concat=False, add_self_loops=False),
            ("flight", "next_rotation", "flight"): GATConv(hidden * heads, hidden, heads=heads, concat=False, add_self_loops=False),
        }, aggr="sum")

        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        # Pass only flight edges to HeteroConv
        flight_x   = {"flight": x_dict["flight"]}
        flight_ei  = {k: v for k, v in edge_index_dict.items()
                      if k[0] == "flight" and k[2] == "flight"}
        h = self.conv1(flight_x, flight_ei)
        h = {k: F.elu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv2(h, flight_ei)
        return self.head(h["flight"]).squeeze(-1)


class FlightHGT(nn.Module):
    """Heterogeneous Graph Transformer using all node types."""

    def __init__(self, in_channels_dict, hidden=64, heads=4, dropout=0.3,
                 metadata=None):
        super().__init__()
        self.dropout = dropout

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
        h = {ntype: F.elu(self.proj[ntype](x)) for ntype, x in x_dict.items()}
        h = self.conv1(h, edge_index_dict)
        h = {k: F.elu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        return self.head(h["flight"]).squeeze(-1)


def build_model(args, graph):
    """Instantiate the requested model from a sample graph's metadata."""
    in_channels_dict = {ntype: graph[ntype].x.shape[1] for ntype in graph.node_types}
    metadata = (graph.node_types, graph.edge_types)

    if args.model == "gat":
        return FlightGAT(
            in_channels=in_channels_dict["flight"],
            hidden=args.hidden,
            heads=args.heads,
            dropout=args.dropout,
        )
    else:  # hgt
        return FlightHGT(
            in_channels_dict=in_channels_dict,
            hidden=args.hidden,
            heads=args.heads,
            dropout=args.dropout,
            metadata=metadata,
        )


#  Training helpers

def train_step(model, graph, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    x_dict  = {k: v.to(device) for k, v in graph.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
    y       = graph["flight"].y.to(device)
    out     = model(x_dict, ei_dict)
    loss    = criterion(out, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def evaluate(model, graph, criterion, device):
    """Returns (mse_loss, mae_minutes)."""
    model.eval()
    with torch.no_grad():
        x_dict  = {k: v.to(device) for k, v in graph.x_dict.items()}
        ei_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
        y_norm  = graph["flight"].y.to(device)
        out     = model(x_dict, ei_dict)
        loss    = criterion(out, y_norm).item()
        y_mean  = graph["flight"].y_mean
        y_std   = graph["flight"].y_std
        pred_min = out.cpu() * y_std + y_mean
        true_min = y_norm.cpu() * y_std + y_mean
        mae      = float((pred_min - true_min).abs().mean())
    return loss, mae


#  Entry point

def parse_args():
    p = argparse.ArgumentParser(description="Iterative per-airport GNN training")
    p.add_argument("--model",     default="gat",  choices=["gat", "hgt"])
    p.add_argument("--epochs",    default=50,     type=int)
    p.add_argument("--lr",        default=1e-3,   type=float)
    p.add_argument("--wd",        default=1e-4,   type=float)
    p.add_argument("--hidden",    default=64,     type=int)
    p.add_argument("--heads",     default=4,      type=int)
    p.add_argument("--dropout",   default=0.3,    type=float)
    p.add_argument("--seed",      default=42,     type=int)
    p.add_argument("--log-every",      default=5,                    type=int)
    p.add_argument("--output",         default="model_iterative_rotation.pt")
    p.add_argument("--wandb-project",  default="tcc-flight-delay")
    return p.parse_args()


def main(args):
    print(f"[config] model={args.model} epochs={args.epochs} seed={args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    wandb.init(
        project=args.wandb_project,
        name=f"train-{args.model}-{datetime.now().strftime('%Y%m%d-%H%M')}",
        config={
            "model":   args.model,
            "epochs":  args.epochs,
            "lr":      args.lr,
            "wd":      args.wd,
            "hidden":  args.hidden,
            "heads":   args.heads,
            "dropout": args.dropout,
        }
    )

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
        model  = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {args.model.upper()} — {n_params:,} params — device={device}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        criterion = nn.MSELoss()

        print("[cache] loading train graphs...")
        train_graphs = []
        for period in train_periods:
            g, _, _ = build_period_graph(loader, *period)
            if g is not None:
                train_graphs.append(g)
        print(f"[cache] {len(train_graphs)} train graphs loaded")

        print("[cache] loading val graphs...")
        val_graphs = []
        for period in val_periods:
            g, _, _ = build_period_graph(loader, *period)
            if g is not None:
                val_graphs.append(g)
        print(f"[cache] {len(val_graphs)} val graphs loaded")

        print("[cache] loading test graphs...")
        test_graphs = []
        for period in test_periods:
            g, _, _ = build_period_graph(loader, *period)
            if g is not None:
                test_graphs.append(g)
        print(f"[cache] {len(test_graphs)} test graphs loaded — starting training\n")

        train_start  = time.time()
        best_val_mae = float("inf")
        best_state   = None
        metrics      = []  

        for epoch in range(1, args.epochs + 1):
            # Train: shuffle indices, iterate cached graphs
            idx_order = list(range(len(train_graphs)))
            random.shuffle(idx_order)
            epoch_train_losses = []

            for i in idx_order:
                loss = train_step(model, train_graphs[i], optimizer, criterion, device)
                epoch_train_losses.append(loss)

            avg_train = float(np.mean(epoch_train_losses)) if epoch_train_losses else 0.0

            # Validate
            val_maes = []
            for g in val_graphs:
                _, mae = evaluate(model, g, criterion, device)
                val_maes.append(mae)

            avg_val_mae = float(np.mean(val_maes)) if val_maes else 0.0
            scheduler.step(avg_val_mae)

            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            metrics.append((epoch, avg_train, avg_val_mae))

            elapsed = time.time() - train_start
            wandb.log({
                "epoch":     epoch,
                "train_mse": avg_train,
                "val_mae":   avg_val_mae,
                "lr":        optimizer.param_groups[0]["lr"],
                "elapsed_s": elapsed,
            })

            if epoch % args.log_every == 0:
                print(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train MSE: {avg_train:.4f} | "
                    f"Val MAE: {avg_val_mae:.1f} min | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                    f"Elapsed: {elapsed:.0f}s"
                )


        model.load_state_dict(best_state)
        test_maes = []
        for g in test_graphs:
            _, mae = evaluate(model, g, criterion, device)
            test_maes.append(mae)

        avg_test_mae = float(np.mean(test_maes)) if test_maes else 0.0
        total_time = time.time() - train_start
        print(f"\nTest MAE (best model): {avg_test_mae:.1f} min | Tempo total: {total_time:.0f}s ({total_time/60:.1f} min)")
        wandb.log({"test_mae": avg_test_mae, "total_train_time_s": total_time})
        wandb.finish()
        del train_graphs, val_graphs, test_graphs


        torch.save(best_state, args.output)
        print(f"Model saved to '{args.output}'")

        import csv
        with open("metrics_iterative_rotation.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_mse", "val_mae"])
            writer.writerows(metrics)
        print("Metrics saved to 'metrics_iterative_rotation.csv'")

    finally:
        loader.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
