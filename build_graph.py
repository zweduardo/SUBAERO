# build_graph.py — Extrai dados do Neo4j e constrói HeteroData para PyG
#
# Subgrafo implementado (fase 2):
#   Nós  : Flight, Airport, Aircraft, Clima
#   Arestas:
#       (Flight)  -[ORIGIN]->          (Airport)
#       (Flight)  -[DESTINATION]->     (Airport)
#       (Flight)  -[NEXT_LEG]->        (Flight)   — mesma rota recorrente
#       (Flight)  -[NEXT_ROTATION]->   (Flight)   — mesmo equipamento no mesmo dia
#       (Flight)  -[ASSIGNED_TO]->     (Aircraft)
#       (Flight)  -[HAS_ORIGIN_WEATHER]-> (Clima)
#       (Clima)   -[OBSERVED_AT]->     (Airport)
#
# Target: delay_arrival_min (regressão) — normalizado em z-score
#
# Uso:
#   python build_graph.py              → salva graph.pt
#   python build_graph.py --stats      → imprime estatísticas do grafo
#
# Dependências:
#   pip install torch torch-geometric neo4j pandas scikit-learn

import argparse
import math
from datetime import datetime, timezone

import torch
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════
import os
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

OUTPUT_PATH    = "graph.pt"   # arquivo salvo

# Top 15 aeroportos brasileiros por volume de tráfego comercial (ANAC)
# Usado pelo flag --filter-airports para reduzir o grafo e economizar RAM.
# Com --since 2025-04-01 resulta em ~355K voos (vs 814K no grafo completo).
MAJOR_AIRPORTS = {
    "SBGR",  # São Paulo – Guarulhos
    "SBSP",  # São Paulo – Congonhas
    "SBKP",  # Campinas – Viracopos
    "SBGL",  # Rio de Janeiro – Galeão
    "SBRJ",  # Rio de Janeiro – Santos Dumont
    "SBBR",  # Brasília
    "SBCF",  # Belo Horizonte – Confins
    "SBSV",  # Salvador
    "SBPA",  # Porto Alegre
    "SBCT",  # Curitiba
    "SBFZ",  # Fortaleza
    "SBRF",  # Recife
    "SBBE",  # Belém
    "SBEG",  # Manaus
    "SBFL",  # Florianópolis
}

# Features usadas por tipo de nó (devem estar no Neo4j)
FLIGHT_FEAT_COLS   = ["hour_dep", "duration_min", "delay_dep_min_clipped"]
AIRPORT_FEAT_COLS  = ["latitude", "longitude", "elevation"]
AIRCRAFT_FEAT_COLS = ["generation_age_years", "is_low_cost"]   # por (airline × tipo)
CLIMA_FEAT_COLS    = ["temp", "windspeed", "rain", "clouds"]

TARGET_COL = "delay_arrival_min"


# ═══════════════════════════════════════════════════════════════════════════
#  Extração do Neo4j
# ═══════════════════════════════════════════════════════════════════════════
class GraphExtractor:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def _run(self, query, **params):
        with self.driver.session() as s:
            return s.run(query, **params).data()

    # ── Nós ────────────────────────────────────────────────────────────────

    def fetch_flights(self, sample=None, since=None, until=None, airports=None,
                      airlines=None, domestic_only=False, international_only=False,
                      equipment_types=None, status_list=None):
        """
        Retorna todos os voos que tem delay_arrival_min definido.
        Se sample for passado, amostra por AERONAVE (preserva cadeias NEXT_LEG)
        ate atingir ~N voos.
        Se airports (set de ICAOs) for passado, filtra voos onde ORIGEM e DESTINO
        estejam ambos no conjunto — reduz RAM ao focar em aeroportos principais.
        """
        print("  Buscando nos Flight...")
        if sample:
            # Amostra por aeronave: pega aeronaves aleatorias e todos os seus voos
            print(f"    Modo PoC: amostrando ~{sample} voos por cadeia de aeronave...")
            rows = self._run(
                """
                MATCH (f:Flight)-[:ASSIGNED_TO]->(ac:Aircraft)
                WHERE f.delay_arrival_min IS NOT NULL
                  AND f.scheduled_departure IS NOT NULL
                  AND f.scheduled_arrival   IS NOT NULL
                RETURN
                    f.flight_id          AS flight_id,
                    f.scheduled_departure AS sched_dep,
                    f.scheduled_arrival   AS sched_arr,
                    f.delay_departure_min AS delay_dep_min,
                    f.delay_arrival_min   AS delay_arrival_min,
                    ac.node_key           AS aircraft_key
                ORDER BY f.flight_id
                """
            )
            df = pd.DataFrame(rows)
            # Conta voos por aeronave e amostra aeronaves ate atingir ~sample voos
            aircraft_list = df["aircraft_key"].unique()
            rng = np.random.RandomState(42)
            rng.shuffle(aircraft_list)
            selected_aircraft = []
            total = 0
            for ac in aircraft_list:
                n_ac = (df["aircraft_key"] == ac).sum()
                selected_aircraft.append(ac)
                total += n_ac
                if total >= sample:
                    break
            df = df[df["aircraft_key"].isin(selected_aircraft)].reset_index(drop=True)
            df = df.drop(columns=["aircraft_key"])
            print(f"    {len(df)} voos de {len(selected_aircraft)} aeronaves selecionadas")
        else:
            where_parts = [
                "f.delay_arrival_min IS NOT NULL",
                "f.scheduled_departure IS NOT NULL",
                "f.scheduled_arrival   IS NOT NULL",
            ]
            params = {}
            if since:
                where_parts.append("f.scheduled_departure >= $since")
                params["since"] = since
            if until:
                where_parts.append("f.scheduled_departure < $until")
                params["until"] = until
            if airports:
                where_parts.append("orig.icao_code IN $airport_list")
                where_parts.append("dest.icao_code IN $airport_list")
                params["airport_list"] = list(airports)
            if airlines:
                where_parts.append("f.airline_code IN $airlines")
                params["airlines"] = list(airlines)
            if domestic_only:
                where_parts.append("f.origin_code STARTS WITH 'SB'")
                where_parts.append("f.destination_code STARTS WITH 'SB'")
            if international_only:
                where_parts.append("NOT (f.origin_code STARTS WITH 'SB' AND f.destination_code STARTS WITH 'SB')")
            if equipment_types:
                where_parts.append("f.equipment_icao IN $equipment_types")
                params["equipment_types"] = list(equipment_types)
            if status_list:
                where_parts.append("f.status IN $status_list")
                params["status_list"] = list(status_list)
            where_clause = "WHERE " + "\n                  AND ".join(where_parts)
            airport_match = (
                "\n                MATCH (f)-[:ORIGIN]->(orig:Airport)"
                "\n                MATCH (f)-[:DESTINATION]->(dest:Airport)"
                if airports else ""
            )
            query = f"""
                MATCH (f:Flight)
                {airport_match}
                {where_clause}
                RETURN
                    f.flight_id          AS flight_id,
                    f.scheduled_departure AS sched_dep,
                    f.scheduled_arrival   AS sched_arr,
                    f.delay_departure_min AS delay_dep_min,
                    f.delay_arrival_min   AS delay_arrival_min
                ORDER BY f.flight_id
                """
            rows = self._run(query, **params)
            df = pd.DataFrame(rows)
            label_parts = []
            if since or until:
                label_parts.append(f"periodo {since} a {until}")
            if airports:
                label_parts.append(f"{len(airports)} aeroportos principais")
            if airlines:
                label_parts.append(f"cias: {','.join(sorted(airlines))}")
            if domestic_only:
                label_parts.append("domestico")
            if international_only:
                label_parts.append("internacional")
            if equipment_types:
                label_parts.append(f"equip: {','.join(sorted(equipment_types))}")
            if status_list:
                label_parts.append(f"status: {','.join(sorted(status_list))}")
            label = f" ({', '.join(label_parts)})" if label_parts else ""
            print(f"    {len(df)} voos com target definido{label}.")
        return df

    def fetch_anchor_flights(self, primary_flight_ids, since):
        """
        Busca voos que são ORIGEM de NEXT_LEG apontando para voos primários,
        mas que estão FORA do período (scheduled_departure < since).
        Esses voos servem como contexto temporal sem serem treinados.
        """
        if not primary_flight_ids:
            return pd.DataFrame(columns=["flight_id", "sched_dep", "sched_arr",
                                         "delay_dep_min", "delay_arrival_min"])
        print("  Buscando voos ancora (predecessores NEXT_LEG)...")
        rows = self._run(
            """
            MATCH (f_prev:Flight)-[:NEXT_LEG]->(f_curr:Flight)
            WHERE f_curr.flight_id IN $primary_ids
              AND f_prev.scheduled_departure < $since
              AND f_prev.delay_arrival_min IS NOT NULL
              AND f_prev.scheduled_departure IS NOT NULL
              AND f_prev.scheduled_arrival   IS NOT NULL
            RETURN DISTINCT
                f_prev.flight_id          AS flight_id,
                f_prev.scheduled_departure AS sched_dep,
                f_prev.scheduled_arrival   AS sched_arr,
                f_prev.delay_departure_min AS delay_dep_min,
                f_prev.delay_arrival_min   AS delay_arrival_min
            """,
            primary_ids=list(primary_flight_ids),
            since=since,
        )
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["flight_id", "sched_dep", "sched_arr",
                                       "delay_dep_min", "delay_arrival_min"])
        print(f"    {len(df)} voos âncora encontrados.")
        return df

    def fetch_airports(self):
        print("  Buscando nós Airport...")
        rows = self._run(
            """
            MATCH (a:Airport)
            RETURN
                a.icao_code   AS icao_code,
                a.latitude    AS latitude,
                a.longitude   AS longitude,
                a.elevation   AS elevation
            ORDER BY a.icao_code
            """
        )
        df = pd.DataFrame(rows)
        print(f"    {len(df)} aeroportos.")
        return df

    def fetch_aircraft(self):
        print("  Buscando nós Aircraft...")
        rows = self._run(
            """
            MATCH (ac:Aircraft)
            WHERE ac.node_key IS NOT NULL
            RETURN
                ac.node_key            AS node_key,
                ac.airline_code        AS airline_code,
                ac.equipment_icao      AS equipment_icao,
                ac.generation_age_years AS generation_age_years,
                ac.is_low_cost         AS is_low_cost
            ORDER BY ac.node_key
            """
        )
        df = pd.DataFrame(rows)
        low = int(df["is_low_cost"].fillna(0).sum()) if "is_low_cost" in df.columns else "?"
        print(f"    {len(df)} Aircraft nodes ({low} low-cost, {len(df)-low} legacy/cargo).")
        return df

    def fetch_clima(self, clima_ids=None):
        print("  Buscando nos Clima...")
        if clima_ids:
            rows = self._run(
                """
                MATCH (c:Clima)
                WHERE c.clima_id IN $ids
                RETURN
                    c.clima_id   AS clima_id,
                    c.temp       AS temp,
                    c.windspeed  AS windspeed,
                    c.rain       AS rain,
                    c.clouds     AS clouds
                ORDER BY c.clima_id
                """,
                ids=list(clima_ids),
            )
        else:
            rows = self._run(
                """
                MATCH (c:Clima)
                RETURN
                    c.clima_id   AS clima_id,
                    c.temp       AS temp,
                    c.windspeed  AS windspeed,
                    c.rain       AS rain,
                    c.clouds     AS clouds
                ORDER BY c.clima_id
                """
            )
        df = pd.DataFrame(rows)
        print(f"    {len(df)} registros de clima.")
        return df

    # ── Arestas ────────────────────────────────────────────────────────────

    def fetch_edges(self, rel_type, src_label, dst_label, src_key, dst_key,
                    filter_ids=None, filter_label=None):
        """
        Busca arestas genéricas retornando par (src_id, dst_id).
        Se filter_ids for passado, filtra arestas onde o no de filter_label
        tem ID na lista (para PoC com subconjunto de voos).
        """
        prop_src = {
            "Flight":   "f.flight_id",
            "Airport":  "a.icao_code",
            "Aircraft": "ac.node_key",
            "Clima":    "c.clima_id",
        }

        aliases = {"Flight": "f", "Airport": "a", "Aircraft": "ac", "Clima": "c"}
        s_alias = aliases[src_label]
        d_alias = aliases[dst_label]
        # Se src e dst sao o mesmo label, usar alias diferente para dst
        if src_label == dst_label:
            d_alias = s_alias + "2"

        src_prop = f"{s_alias}.{prop_src[src_label].split('.')[1]}"
        dst_prop = f"{d_alias}.{prop_src[dst_label].split('.')[1]}"

        where_clause = ""
        params = {}
        if filter_ids is not None and filter_label:
            # Filtrar pelo source
            filter_alias = s_alias if filter_label == src_label else d_alias
            filter_field = prop_src[filter_label].split('.')[1]
            where_clause = f"WHERE {filter_alias}.{filter_field} IN $filter_ids"
            params["filter_ids"] = list(filter_ids)

        query = f"""
            MATCH ({s_alias}:{src_label})-[:{rel_type}]->({d_alias}:{dst_label})
            {where_clause}
            RETURN {src_prop} AS src_id,
                   {dst_prop} AS dst_id
        """
        print(f"  Buscando arestas {src_label}-[{rel_type}]->{dst_label}...")
        rows = self._run(query, **params)
        df = pd.DataFrame(rows, columns=["src_id", "dst_id"])
        print(f"    {len(df)} arestas.")
        return df


# ═══════════════════════════════════════════════════════════════════════════
#  Feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def _parse_iso(s):
    """Converte string ISO 8601 em datetime (retorna None em falha)."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return None


def build_flight_features(df):
    """
    Gera features numéricas para cada voo:
      - hour_dep            : hora UTC da partida prevista (0–23)
      - duration_min        : duração planejada em minutos
      - delay_dep_min_clipped: atraso na partida clipado em [-30, 300]
    """
    sched_dep = df["sched_dep"].apply(_parse_iso)
    sched_arr = df["sched_arr"].apply(_parse_iso)

    df = df.copy()
    df["hour_dep"] = sched_dep.apply(
        lambda dt: dt.hour if dt is not None else float("nan")
    )
    df["duration_min"] = [
        (arr - dep).total_seconds() / 60.0
        if dep is not None and arr is not None else float("nan")
        for dep, arr in zip(sched_dep, sched_arr)
    ]
    df["delay_dep_min_clipped"] = df["delay_dep_min"].clip(-30, 300)

    # Preenche NaN com mediana
    for col in FLIGHT_FEAT_COLS:
        med = df[col].median()
        df[col] = df[col].fillna(med)

    return df


def build_airport_features(df):
    df = df.copy()
    for col in AIRPORT_FEAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def build_aircraft_features(df):
    """
    Gera features numéricas para cada Aircraft node (airline × tipo):
      - generation_age_years : idade média do tipo no Brasil (ANAC RAB)
      - is_low_cost          : 1.0 se low-cost, 0.0 se legacy/cargo
    """
    df = df.copy()
    for col in AIRCRAFT_FEAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "generation_age_years" in df.columns:
        med = df["generation_age_years"].median()
        df["generation_age_years"] = df["generation_age_years"].fillna(med if not pd.isna(med) else 15.0)
    if "is_low_cost" in df.columns:
        df["is_low_cost"] = df["is_low_cost"].fillna(0.0)
    return df


def build_clima_features(df):
    df = df.copy()
    for col in CLIMA_FEAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def scale_features(df, cols, scaler=None):
    """Z-score normaliza colunas; retorna (array, scaler)."""
    if not cols:
        # Retorna tensor de zeros (1 feature dummy) se sem features
        dummy = np.zeros((len(df), 1), dtype=np.float32)
        return dummy, None
    X = df[cols].values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler


# ═══════════════════════════════════════════════════════════════════════════
#  Construção do HeteroData
# ═══════════════════════════════════════════════════════════════════════════

def build_hetero_graph(extractor: GraphExtractor, sample=None, since=None, until=None, airports=None,
                       airlines=None, domestic_only=False, international_only=False,
                       equipment_types=None, status_list=None) -> HeteroData:
    print("\n[1/3] Extraindo nos do Neo4j...")
    flights_df = extractor.fetch_flights(sample=sample, since=since, until=until, airports=airports,
                                         airlines=airlines, domestic_only=domestic_only,
                                         international_only=international_only,
                                         equipment_types=equipment_types, status_list=status_list)
    primary_flight_ids = set(flights_df["flight_id"])

    # Busca ancoras NEXT_LEG quando filtrando por periodo
    if since and not sample:
        anchor_df = extractor.fetch_anchor_flights(primary_flight_ids, since)
        if not anchor_df.empty:
            flights_df = pd.concat([flights_df, anchor_df], ignore_index=True)
            print(f"    Total com ancoras: {len(flights_df)} voos")
    else:
        primary_flight_ids = None  # sem filtro = todos são primários

    airports_df = extractor.fetch_airports()
    aircraft_df = extractor.fetch_aircraft()

    # IDs dos voos selecionados (para filtrar arestas no Neo4j quando em modo sample)
    flight_ids = set(flights_df["flight_id"]) if (sample or since) else None

    print("\n[2/3] Extraindo arestas do Neo4j...")
    flt = ("Flight", flight_ids) if flight_ids else (None, None)
    origin_edges      = extractor.fetch_edges("ORIGIN",             "Flight",  "Airport",  "flight_id", "icao_code",
                                              filter_ids=flt[1], filter_label=flt[0])
    dest_edges        = extractor.fetch_edges("DESTINATION",        "Flight",  "Airport",  "flight_id", "icao_code",
                                              filter_ids=flt[1], filter_label=flt[0])
    next_leg_edges    = extractor.fetch_edges("NEXT_LEG",           "Flight",  "Flight",   "flight_id", "flight_id",
                                              filter_ids=flt[1], filter_label=flt[0])
    next_rot_edges    = extractor.fetch_edges("NEXT_ROTATION",      "Flight",  "Flight",   "flight_id", "flight_id",
                                              filter_ids=flt[1], filter_label=flt[0])
    assigned_edges    = extractor.fetch_edges("ASSIGNED_TO",        "Flight",  "Aircraft", "flight_id", "node_key",
                                              filter_ids=flt[1], filter_label=flt[0])
    has_weather_edges = extractor.fetch_edges("HAS_ORIGIN_WEATHER", "Flight",  "Clima",    "flight_id", "clima_id",
                                              filter_ids=flt[1], filter_label=flt[0])

    # Carrega apenas os climas referenciados pelos voos selecionados
    clima_ids_needed = set(has_weather_edges["dst_id"]) if len(has_weather_edges) > 0 else set()
    if clima_ids_needed:
        clima_df = extractor.fetch_clima(clima_ids=clima_ids_needed)
    else:
        clima_df = extractor.fetch_clima()

    # OBSERVED_AT: filtra pelos climas que realmente temos
    clima_id_set = set(clima_df["clima_id"])
    observed_edges    = extractor.fetch_edges("OBSERVED_AT",        "Clima",   "Airport",  "clima_id",  "icao_code",
                                              filter_ids=clima_id_set, filter_label="Clima")

    print("\n[3/3] Construindo HeteroData...")

    # ── Índices locais (string → int) ──────────────────────────────────
    flight_idx   = {fid: i for i, fid in enumerate(flights_df["flight_id"])}
    airport_idx  = {code: i for i, code in enumerate(airports_df["icao_code"])}
    aircraft_idx = {key: i for i, key in enumerate(aircraft_df["node_key"])}
    clima_idx    = {cid: i for i, cid in enumerate(clima_df["clima_id"])}

    def edge_tensor(df_edges, src_map, dst_map):
        """Converte DataFrame (src_id, dst_id) em tensor [2, E] filtrando IDs desconhecidos.

        Usa numpy array lookup (10-20x mais rapido que iterrows, sem dependencia de
        versao do pandas para Series.map com dict).
        """
        src_col = df_edges["src_id"].to_numpy()
        dst_col = df_edges["dst_id"].to_numpy()
        src_ids = np.array([src_map.get(x, -1) for x in src_col], dtype="int64")
        dst_ids = np.array([dst_map.get(x, -1) for x in dst_col], dtype="int64")
        mask = (src_ids >= 0) & (dst_ids >= 0)
        skipped = int((~mask).sum())
        if skipped:
            print(f"      {skipped} arestas ignoradas (nó fora do subconjunto).")
        if mask.sum() == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.from_numpy(np.stack([src_ids[mask], dst_ids[mask]]))

    # ── Feature engineering ─────────────────────────────────────────────
    flights_df  = build_flight_features(flights_df)
    airports_df = build_airport_features(airports_df)
    aircraft_df = build_aircraft_features(aircraft_df)
    clima_df    = build_clima_features(clima_df)

    flight_X,  _  = scale_features(flights_df,  FLIGHT_FEAT_COLS)
    airport_X, _  = scale_features(airports_df, AIRPORT_FEAT_COLS)
    aircraft_X, _ = scale_features(aircraft_df, AIRCRAFT_FEAT_COLS)
    clima_X, _    = scale_features(clima_df,     CLIMA_FEAT_COLS)

    # ── Target (z-score) ────────────────────────────────────────────────
    y_raw = flights_df[TARGET_COL].values.astype(np.float32)
    # Clip extremos antes de normalizar (outliers de >6h são raros e ruidosos)
    y_clipped = np.clip(y_raw, -30, 360)
    y_mean, y_std = float(y_clipped.mean()), float(y_clipped.std()) + 1e-6
    y_norm = (y_clipped - y_mean) / y_std

    # ── Montar HeteroData ───────────────────────────────────────────────
    data = HeteroData()

    # Nós
    data["flight"].x  = torch.tensor(flight_X,   dtype=torch.float)
    data["flight"].y  = torch.tensor(y_norm,      dtype=torch.float)
    data["flight"].y_raw  = torch.tensor(y_raw,   dtype=torch.float)  # para denormalizar
    data["flight"].y_mean = y_mean
    data["flight"].y_std  = y_std
    data["flight"].flight_ids = list(flights_df["flight_id"])   # para debug

    # new_mask: True para voos primarios (no periodo), False para ancoras
    if primary_flight_ids is not None:
        new_mask_list = [fid in primary_flight_ids for fid in flights_df["flight_id"]]
        data["flight"].new_mask = torch.tensor(new_mask_list, dtype=torch.bool)
        n_primary = sum(new_mask_list)
        n_anchor  = len(new_mask_list) - n_primary
        print(f"    new_mask: {n_primary} voos primarios + {n_anchor} ancoras")

    data["airport"].x   = torch.tensor(airport_X,  dtype=torch.float)
    data["airport"].icao = list(airports_df["icao_code"])

    data["aircraft"].x  = torch.tensor(aircraft_X, dtype=torch.float)
    data["aircraft"].node_keys      = list(aircraft_df["node_key"])
    data["aircraft"].equipment_icao = list(aircraft_df["equipment_icao"])

    data["clima"].x     = torch.tensor(clima_X,    dtype=torch.float)
    data["clima"].clima_ids = list(clima_df["clima_id"])

    # Arestas
    data["flight", "ORIGIN",             "airport"].edge_index  = edge_tensor(origin_edges,      flight_idx, airport_idx)
    data["flight", "DESTINATION",        "airport"].edge_index  = edge_tensor(dest_edges,         flight_idx, airport_idx)
    data["flight", "NEXT_LEG",           "flight"].edge_index   = edge_tensor(next_leg_edges,     flight_idx, flight_idx)
    data["flight", "NEXT_ROTATION",      "flight"].edge_index   = edge_tensor(next_rot_edges,     flight_idx, flight_idx)
    data["flight", "ASSIGNED_TO",        "aircraft"].edge_index = edge_tensor(assigned_edges,     flight_idx, aircraft_idx)
    data["flight", "HAS_ORIGIN_WEATHER", "clima"].edge_index    = edge_tensor(has_weather_edges,  flight_idx, clima_idx)
    data["clima",  "OBSERVED_AT",        "airport"].edge_index  = edge_tensor(observed_edges,     clima_idx,  airport_idx)

    data = ToUndirected()(data)
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  Stats helper
# ═══════════════════════════════════════════════════════════════════════════

def print_stats(data: HeteroData):
    print("\n------------ Estatisticas do Grafo ------------")
    print("Nos:")
    for ntype in data.node_types:
        x = data[ntype].get("x")
        n = x.shape[0] if x is not None else "?"
        f = x.shape[1] if x is not None else "?"
        print(f"  {ntype:12s}  {n:>8} nos  x  {f} features")

    y = data["flight"].y
    print(f"\n  Target (flight.y) - normalizado:")
    print(f"    min={y.min():.2f}  max={y.max():.2f}  "
          f"mean={y.mean():.4f}  std={y.std():.4f}")
    y_raw = data["flight"].y_raw
    print(f"  Target (flight.y_raw) - minutos brutos:")
    print(f"    min={y_raw.min():.1f}  max={y_raw.max():.1f}  "
          f"mean={y_raw.mean():.1f}  std={y_raw.std():.1f}")

    print("\nArestas:")
    for etype in data.edge_types:
        ei = data[etype].edge_index
        print(f"  {str(etype):55s}  {ei.shape[1]:>8} arestas")

    # Proporcao de voos atrasados (>15 min)
    delayed = (y_raw > 15).sum().item()
    pct = 100 * delayed / len(y_raw)
    print(f"\n  Voos com atraso > 15 min: {delayed}/{len(y_raw)} ({pct:.1f}%)")
    print("-----------------------------------------------\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true",
                        help="Imprime estatísticas do grafo após construção")
    parser.add_argument("--output", default=OUTPUT_PATH,
                        help=f"Caminho de saída (padrão: {OUTPUT_PATH})")
    parser.add_argument("--sample", type=int, default=None,
                        help="Amostrar N voos (para PoC rápida, ex: --sample 50000)")
    parser.add_argument("--since", type=str, default=None,
                        help="Data inicial inclusiva (YYYY-MM-DD), ex: --since 2025-01-01")
    parser.add_argument("--until", type=str, default=None,
                        help="Data final exclusiva (YYYY-MM-DD), ex: --until 2025-07-01")
    parser.add_argument("--filter-airports", action="store_true",
                        help="Filtra voos onde origem E destino estao em MAJOR_AIRPORTS (~25 principais BR)")
    parser.add_argument("--airline", type=str, default=None,
                        help="Companhias aereas ICAO separadas por virgula (ex: GLO,TAM,AZU)")
    parser.add_argument("--domestic-only", action="store_true",
                        help="Somente voos domesticos (origem e destino com prefixo ICAO 'SB')")
    parser.add_argument("--international-only", action="store_true",
                        help="Somente voos internacionais (pelo menos um aeroporto fora do Brasil)")
    parser.add_argument("--equipment", type=str, default=None,
                        help="Tipos de aeronave ICAO separados por virgula (ex: B738,A320)")
    parser.add_argument("--status", type=str, default=None,
                        help="Status dos voos separados por virgula (ex: Realizado,Cancelado)")
    args = parser.parse_args()

    if args.domestic_only and args.international_only:
        parser.error("--domestic-only e --international-only sao mutuamente exclusivos.")

    airports        = MAJOR_AIRPORTS if args.filter_airports else None
    airlines        = set(args.airline.split(",")) if args.airline else None
    equipment_types = set(args.equipment.split(",")) if args.equipment else None
    status_list     = set(args.status.split(",")) if args.status else None

    extractor = GraphExtractor()
    try:
        data = build_hetero_graph(extractor, sample=args.sample,
                                  since=args.since, until=args.until,
                                  airports=airports, airlines=airlines,
                                  domestic_only=args.domestic_only,
                                  international_only=args.international_only,
                                  equipment_types=equipment_types,
                                  status_list=status_list)
    finally:
        extractor.close()

    if args.stats:
        print_stats(data)

    torch.save(data, args.output)
    print(f"Grafo salvo em '{args.output}'")
    print(f"  flight.x  : {data['flight'].x.shape}")
    print(f"  flight.y  : {data['flight'].y.shape}")
    print(f"Pronto! Para carregar: data = torch.load('{args.output}')")