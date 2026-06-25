# predict.py — Preve atraso de chegada para voos futuros
#
# Uso:
#   python predict.py --graph graph.pt --model model.pt --model-type gat
#   python predict.py --graph graph.pt --model model.pt --model-type hgt --flights "FLT001,FLT002"
#   python predict.py --graph graph.pt --model model.pt --model-type gat --future-since 2026-03-20
#
# Modos:
#   1. --flights: preve para flight_ids especificos ja no grafo
#   2. --future-since: busca voos futuros no Neo4j (sem delay_arrival_min),
#      adiciona temporariamente ao grafo e preve
#   3. Sem flags: preve para TODOS os voos do grafo (util para analise)
#
# Para voos futuros, delay_departure_min e desconhecido -> usa 0.
# A GNN infere o atraso provavel via NEXT_LEG (propagacao da aeronave anterior)
# e via clima/aeroporto.

import argparse
import sys

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from build_graph import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    FLIGHT_FEAT_COLS, CLIMA_FEAT_COLS,
    GraphExtractor, build_flight_features, build_clima_features,
    scale_features,
)
from train import build_model


def load_model(model_path, model_type, data):
    """Carrega modelo treinado."""
    in_channels_dict = {nt: data[nt].x.shape[1] for nt in data.node_types}
    model = build_model(
        name=model_type,
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        hidden=64,
        heads=4,
        dropout=0.0,  # sem dropout na inferencia
    )
    # Forward dummy para inicializar lazy params
    with torch.no_grad():
        model.eval()
        _ = model(data.x_dict, data.edge_index_dict)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def add_future_flights(data, since_date):
    """
    Busca voos futuros do Neo4j (sem delay_arrival_min) e adiciona ao grafo.
    Retorna (data_atualizado, future_indices, future_flight_ids).
    """
    extractor = GraphExtractor()
    try:
        print(f"Buscando voos futuros (desde {since_date})...")
        rows = extractor._run(
            """
            MATCH (f:Flight)
            WHERE f.delay_arrival_min IS NULL
              AND f.scheduled_departure IS NOT NULL
              AND f.scheduled_arrival   IS NOT NULL
              AND f.scheduled_departure >= $since
            RETURN
                f.flight_id          AS flight_id,
                f.scheduled_departure AS sched_dep,
                f.scheduled_arrival   AS sched_arr,
                f.delay_departure_min AS delay_dep_min
            ORDER BY f.scheduled_departure
            """,
            since=since_date,
        )
        future_df = pd.DataFrame(rows)
        print(f"  {len(future_df)} voos futuros encontrados")

        if len(future_df) == 0:
            return data, [], []

        # Filtrar voos ja no grafo
        existing_ids = set(data["flight"].flight_ids)
        future_df = future_df[~future_df["flight_id"].isin(existing_ids)].reset_index(drop=True)
        print(f"  {len(future_df)} sao novos (nao estao no grafo)")

        if len(future_df) == 0:
            return data, [], []

        future_flight_ids = list(future_df["flight_id"])
        future_ids_set = set(future_flight_ids)

        # delay_arrival_min = 0 (placeholder, sera previsto)
        future_df["delay_arrival_min"] = 0.0
        # delay_dep_min: usa 0 se nulo (voo futuro)
        future_df["delay_dep_min"] = future_df["delay_dep_min"].fillna(0.0)

        # Buscar arestas
        origin_e = extractor.fetch_edges(
            "ORIGIN", "Flight", "Airport", "flight_id", "icao_code",
            filter_ids=future_ids_set, filter_label="Flight")
        dest_e = extractor.fetch_edges(
            "DESTINATION", "Flight", "Airport", "flight_id", "icao_code",
            filter_ids=future_ids_set, filter_label="Flight")
        next_leg_e = extractor.fetch_edges(
            "NEXT_LEG", "Flight", "Flight", "flight_id", "flight_id",
            filter_ids=future_ids_set, filter_label="Flight")
        assigned_e = extractor.fetch_edges(
            "ASSIGNED_TO", "Flight", "Aircraft", "flight_id", "equipment_icao",
            filter_ids=future_ids_set, filter_label="Flight")
        weather_e = extractor.fetch_edges(
            "HAS_ORIGIN_WEATHER", "Flight", "Clima", "flight_id", "clima_id",
            filter_ids=future_ids_set, filter_label="Flight")

        # Novos climas
        old_clima_ids = set(data["clima"].clima_ids) if hasattr(data["clima"], "clima_ids") else set()
        needed_clima = set(weather_e["dst_id"]) - old_clima_ids
        if needed_clima:
            new_clima_df = extractor.fetch_clima(clima_ids=needed_clima)
            new_observed = extractor.fetch_edges(
                "OBSERVED_AT", "Clima", "Airport", "clima_id", "icao_code",
                filter_ids=needed_clima, filter_label="Clima")
        else:
            new_clima_df = pd.DataFrame(columns=["clima_id", "temp", "windspeed", "rain", "clouds"])
            new_observed = pd.DataFrame(columns=["src_id", "dst_id"])
    finally:
        extractor.close()

    # -- Feature engineering
    future_df = build_flight_features(future_df)
    future_X, _ = scale_features(future_df, FLIGHT_FEAT_COLS)

    y_mean = data["flight"].y_mean
    y_std = data["flight"].y_std

    # -- Indices
    n_old = data["flight"].x.shape[0]
    n_old_clima = data["clima"].x.shape[0]

    airport_idx = {code: i for i, code in enumerate(data["airport"].icao)}
    aircraft_idx = {}
    if hasattr(data["aircraft"], "equipment_icao"):
        aircraft_idx = {eq: i for i, eq in enumerate(data["aircraft"].equipment_icao)}

    all_flight_ids = list(data["flight"].flight_ids) + future_flight_ids
    full_flight_idx = {fid: i for i, fid in enumerate(all_flight_ids)}

    all_clima_ids = list(data["clima"].clima_ids) if hasattr(data["clima"], "clima_ids") else []
    for i, cid in enumerate(new_clima_df["clima_id"] if len(new_clima_df) > 0 else []):
        all_clima_ids.append(cid)
    full_clima_idx = {cid: i for i, cid in enumerate(all_clima_ids)}

    def edge_tensor(df_edges, src_map, dst_map):
        src_list, dst_list = [], []
        for _, row in df_edges.iterrows():
            s = src_map.get(row["src_id"])
            d = dst_map.get(row["dst_id"])
            if s is not None and d is not None:
                src_list.append(s)
                dst_list.append(d)
        if not src_list:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor([src_list, dst_list], dtype=torch.long)

    def append_edges(edge_type, new_ei):
        if new_ei.shape[1] == 0:
            return
        old_ei = data[edge_type].edge_index
        data[edge_type].edge_index = torch.cat([old_ei, new_ei], dim=1)

    # -- Append nos
    data["flight"].x = torch.cat([
        data["flight"].x,
        torch.tensor(future_X, dtype=torch.float)
    ], dim=0)
    # y dummy (sera substituido pela previsao)
    data["flight"].y = torch.cat([
        data["flight"].y,
        torch.zeros(len(future_df), dtype=torch.float)
    ])
    data["flight"].y_raw = torch.cat([
        data["flight"].y_raw,
        torch.zeros(len(future_df), dtype=torch.float)
    ])
    data["flight"].flight_ids = all_flight_ids

    if len(new_clima_df) > 0:
        new_clima_df = build_clima_features(new_clima_df)
        new_clima_X, _ = scale_features(new_clima_df, CLIMA_FEAT_COLS)
        data["clima"].x = torch.cat([
            data["clima"].x,
            torch.tensor(new_clima_X, dtype=torch.float)
        ], dim=0)
    if hasattr(data["clima"], "clima_ids"):
        data["clima"].clima_ids = all_clima_ids

    # -- Append arestas
    append_edges(("flight", "ORIGIN", "airport"),
                 edge_tensor(origin_e, full_flight_idx, airport_idx))
    append_edges(("flight", "DESTINATION", "airport"),
                 edge_tensor(dest_e, full_flight_idx, airport_idx))
    append_edges(("flight", "NEXT_LEG", "flight"),
                 edge_tensor(next_leg_e, full_flight_idx, full_flight_idx))
    append_edges(("flight", "ASSIGNED_TO", "aircraft"),
                 edge_tensor(assigned_e, full_flight_idx, aircraft_idx))
    append_edges(("flight", "HAS_ORIGIN_WEATHER", "clima"),
                 edge_tensor(weather_e, full_flight_idx, full_clima_idx))
    append_edges(("clima", "OBSERVED_AT", "airport"),
                 edge_tensor(new_observed, full_clima_idx, airport_idx))

    future_indices = list(range(n_old, n_old + len(future_df)))
    return data, future_indices, future_flight_ids


@torch.no_grad()
def predict(model, data, indices, flight_ids):
    """Roda inferencia e retorna previsoes em minutos."""
    model.eval()
    pred_norm = model(data.x_dict, data.edge_index_dict)

    y_mean = data["flight"].y_mean
    y_std = data["flight"].y_std

    results = []
    for idx, fid in zip(indices, flight_ids):
        pred_min = pred_norm[idx].item() * y_std + y_mean
        actual_raw = data["flight"].y_raw[idx].item()
        results.append({
            "flight_id": fid,
            "predicted_delay_min": round(pred_min, 1),
            "actual_delay_min": round(actual_raw, 1) if actual_raw != 0.0 else None,
        })

    return pd.DataFrame(results)


def main(args):
    print(f"Carregando grafo '{args.graph}'...")
    data = torch.load(args.graph)
    data = ToUndirected()(data)

    future_indices = []
    future_ids = []

    # Modo 1: voos futuros do Neo4j
    if args.future_since:
        data, future_indices, future_ids = add_future_flights(data, args.future_since)
        if not future_indices:
            print("Nenhum voo futuro encontrado.")
            return

    print(f"\nCarregando modelo '{args.model}' (tipo: {args.model_type})...")
    model = load_model(args.model, args.model_type, data)

    # Determinar quais voos prever
    if args.flights:
        # Modo 2: flight_ids especificos
        target_ids = [f.strip() for f in args.flights.split(",")]
        id_to_idx = {fid: i for i, fid in enumerate(data["flight"].flight_ids)}
        indices = []
        ids = []
        for fid in target_ids:
            if fid in id_to_idx:
                indices.append(id_to_idx[fid])
                ids.append(fid)
            else:
                print(f"  AVISO: {fid} nao encontrado no grafo")
    elif future_indices:
        indices = future_indices
        ids = future_ids
    else:
        # Modo 3: todos os voos
        n = data["flight"].x.shape[0]
        indices = list(range(n))
        ids = list(data["flight"].flight_ids)

    print(f"\nPrevendo {len(indices)} voos...")
    results = predict(model, data, indices, ids)

    # Mostrar resultados
    if len(results) <= 50:
        print("\n" + results.to_string(index=False))
    else:
        print(f"\nPrimeiros 20 resultados:")
        print(results.head(20).to_string(index=False))
        print(f"... ({len(results)} voos no total)")

    # Salvar CSV
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nPrevisoes salvas em '{args.output}'")

    # Stats
    preds = results["predicted_delay_min"]
    print(f"\nEstatisticas das previsoes:")
    print(f"  Media: {preds.mean():.1f} min")
    print(f"  Mediana: {preds.median():.1f} min")
    print(f"  Min: {preds.min():.1f} min  |  Max: {preds.max():.1f} min")
    print(f"  Voos com atraso previsto > 15min: {(preds > 15).sum()}/{len(preds)}")

    if results["actual_delay_min"].notna().any():
        mask = results["actual_delay_min"].notna()
        actual = results.loc[mask, "actual_delay_min"]
        pred = results.loc[mask, "predicted_delay_min"]
        mae = (pred - actual).abs().mean()
        print(f"\n  MAE (voos com valor real): {mae:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preve atraso de chegada de voos")
    parser.add_argument("--graph", default="graph.pt", help="Grafo de entrada")
    parser.add_argument("--model", default="model.pt", help="Pesos do modelo")
    parser.add_argument("--model-type", default="gat", choices=["gat", "hgt", "tgn"],
                        dest="model_type", help="Arquitetura do modelo")
    parser.add_argument("--flights", default=None,
                        help="Flight IDs para prever (separados por virgula)")
    parser.add_argument("--future-since", default=None, dest="future_since",
                        help="Busca voos futuros no Neo4j desde esta data (ISO)")
    parser.add_argument("--output", default=None,
                        help="Salvar previsoes em CSV")
    args = parser.parse_args()
    main(args)
