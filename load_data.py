# load_data.py — Interação com o Neo4j
#
# Modelo do Grafo (4 camadas):
#   1. Infraestrutura  →  (Airport)  nó com lat/lon, ICAO, IATA …
#   2. Voo             →  (Flight)   nó com data, airline, status, delay …
#                          (Flight)-[:ORIGIN]->(Airport)
#                          (Flight)-[:DESTINATION]->(Airport)
#   3. Rota            →  (Flight)-[:NEXT_LEG]->(Flight)
#                          (Aircraft)  nó com tail_number/modelo
#                          (Flight)-[:ASSIGNED_TO]->(Aircraft)
#   4. Ambiental       →  (Clima)  nó com dados meteorológicos
#                          (Clima)-[:OBSERVED_AT]->(Airport)
#                          (Flight)-[:HAS_CONDITIONS]->(Clima)

import pandas as pd
from neo4j import GraphDatabase
from datetime import datetime, timezone, timedelta
import json
import time as tm

from api_calls import (
    get_vra_data, get_clima, get_clima_historico, tratar_clima,
    get_clima_openmeteo,
    load_opensky_credentials, get_opensky_flights_by_time,
    get_opensky_aircraft_metadata, get_opensky_aircraft_age, get_opensky_def,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Classe Neo4j — encapsula toda interação com o banco
# ═══════════════════════════════════════════════════════════════════════════
class Neo4jDB:
    def __init__(self, uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                 user="neo4j", password="tcc12345"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # ── Limpeza ────────────────────────────────────────────────────────────
    def clear_all(self):
        """Remove TODOS os nós e relacionamentos do banco."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Banco Neo4j limpo.")

    # ── Constraints / Índices (rodar uma vez) ──────────────────────────────
    def create_constraints(self):
        """Cria constraints de unicidade para acelerar MATCH."""
        with self.driver.session() as session:
            # Dropar constraint antiga de iata_code (não é sempre único)
            try:
                result = session.run("SHOW CONSTRAINTS")
                for record in result:
                    name = record.get("name", "")
                    props = record.get("properties", [])
                    label = record.get("labelsOrTypes", [])
                    if "iata_code" in props and "Airport" in label:
                        session.run(f"DROP CONSTRAINT {name}")
                        print(f"  Constraint '{name}' removida (iata_code).")
            except Exception:
                pass

        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Airport)   REQUIRE a.icao_code IS UNIQUE", #Unico icao_code
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ac:Aircraft)  REQUIRE ac.node_key IS UNIQUE", #Unico node_key (airline_code + equipment_icao)
            "CREATE INDEX IF NOT EXISTS FOR (a:Airport) ON (a.iata_code)", #Indicies para acelerar buscas
            "CREATE INDEX IF NOT EXISTS FOR (f:Flight) ON (f.flight_id)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Flight) ON (f.scheduled_departure)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Clima) ON (c.clima_id)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Flight) ON (f.airline_code, f.flight_number)",
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception:
                    pass  # já existe

    # ── Airport ────────────────────────────────────────────────────────────
    def create_airport(self, type, name, latitude, longitude, elevation,
                       iso_country, iso_region, municipality,
                       icao_code, iata_code, home_link):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (a:Airport {icao_code: $icao_code})
                ON CREATE SET
                    a.type = $type, a.name = $name,
                    a.latitude = $latitude, a.longitude = $longitude,
                    a.elevation = $elevation, a.iso_country = $iso_country,
                    a.iso_region = $iso_region, a.municipality = $municipality,
                    a.iata_code = $iata_code, a.home_link = $home_link
                """,
                type=type, name=name,
                latitude=latitude, longitude=longitude,
                elevation=elevation, iso_country=iso_country,
                iso_region=iso_region, municipality=municipality,
                icao_code=icao_code, iata_code=iata_code,
                home_link=home_link,
            )

    # ── Flight batch — UNWIND para inserir centenas de voos por transação ──
    def create_flights_batch(self, flights_batch):
        """
        Insere um lote de voos usando UNWIND (muito mais rápido que 1-a-1).
        flights_batch: lista de dicts com chaves padronizadas.
        """
        if not flights_batch:
            return
        with self.driver.session() as session:
            session.run(
                """
                UNWIND $rows AS r
                MATCH (orig:Airport)
                    WHERE orig.icao_code = r.origin_code
                       OR orig.iata_code = r.origin_code
                MATCH (dest:Airport)
                    WHERE dest.icao_code = r.destination_code
                       OR dest.iata_code = r.destination_code
                MERGE (f:Flight {flight_id: r.flight_id})
                ON CREATE SET
                    f.date = r.date, f.airline_code = r.airline_code,
                    f.flight_number = r.flight_number, f.status = r.status,
                    f.origin_code = r.origin_code,
                    f.scheduled_departure = r.scheduled_dep,
                    f.actual_departure    = r.actual_dep,
                    f.scheduled_arrival   = r.scheduled_arr,
                    f.actual_arrival      = r.actual_arr,
                    f.delay_departure_min = r.delay_dep_min,
                    f.delay_arrival_min   = r.delay_arr_min,
                    f.equipment_icao      = r.equipment_icao
                MERGE (f)-[:ORIGIN]->(orig)
                MERGE (f)-[:DESTINATION]->(dest)
                """,
                rows=flights_batch,
            )

    # ── Aircraft + ASSIGNED_TO ─────────────────────────────────────────────
    def create_aircraft_and_assign_by_airline(self, airline_code, equipment_icao):
        """Cria nó Aircraft por (airline × tipo) e liga os Flights correspondentes."""
        if not equipment_icao or not airline_code:
            return
        node_key = f"{airline_code}_{equipment_icao}"
        with self.driver.session() as session:
            session.run(
                """
                MERGE (ac:Aircraft {node_key: $node_key})
                ON CREATE SET ac.airline_code  = $airline,
                              ac.equipment_icao = $equip,
                              ac.node_key       = $node_key
                WITH ac
                MATCH (f:Flight {airline_code: $airline, equipment_icao: $equip})
                MERGE (f)-[:ASSIGNED_TO]->(ac)
                """,
                node_key=node_key, airline=airline_code, equip=equipment_icao,
            )

    # ── OpenSky — cria Aircraft individuais e atualiza Flight ────────────
    def update_flights_individual_aircraft_batch(self, rows: list[dict]):
        """
        Para cada item {flight_id, icao24, registration, aircraft_age_years, equipment_icao}:
          - SET Flight.icao24
          - MERGE Aircraft(icao24) com registration, age, node_key
          - Redireciona ASSIGNED_TO para o Aircraft individual
        """
        if not rows:
            return
        with self.driver.session() as session:
            session.run(
                """
                UNWIND $rows AS r
                MATCH (f:Flight {flight_id: r.flight_id})
                SET f.icao24 = r.icao24

                MERGE (ac_new:Aircraft {icao24: r.icao24})
                ON CREATE SET
                    ac_new.registration       = r.registration,
                    ac_new.aircraft_age_years = r.aircraft_age_years,
                    ac_new.equipment_icao     = r.equipment_icao,
                    ac_new.node_key           = r.icao24
                ON MATCH SET
                    ac_new.registration       = coalesce(ac_new.registration, r.registration),
                    ac_new.aircraft_age_years = coalesce(ac_new.aircraft_age_years, r.aircraft_age_years),
                    ac_new.equipment_icao     = coalesce(ac_new.equipment_icao, r.equipment_icao),
                    ac_new.node_key           = r.icao24

                WITH f, ac_new
                OPTIONAL MATCH (f)-[old:ASSIGNED_TO]->(:Aircraft)
                DELETE old
                MERGE (f)-[:ASSIGNED_TO]->(ac_new)
                """,
                rows=rows,
            )

    # ── NEXT_LEG — liga voos consecutivos na mesma rota/companhia ────────
    def create_next_legs(self):
        """
        Cria relações NEXT_LEG entre voos consecutivos do mesmo
        número de voo (mesma airline + flight_number).
        Processa por grupo (airline, flight_number) para evitar cartesian join.
        """
        with self.driver.session() as session:
            # Busca todas as combinações (airline, flight_number) distintas
            result = session.run(
                """
                MATCH (f:Flight)
                WHERE f.scheduled_departure IS NOT NULL
                RETURN DISTINCT f.airline_code AS airline, f.flight_number AS fnum
                """
            )
            groups = [(r["airline"], r["fnum"]) for r in result]

        print(f"  {len(groups)} grupos (airline, flight_number) para NEXT_LEG...")
        created = 0
        for airline, fnum in groups:
            with self.driver.session() as session:
                res = session.run(
                    """
                    MATCH (f:Flight)
                    WHERE f.airline_code = $airline
                      AND f.flight_number = $fnum
                      AND f.scheduled_departure IS NOT NULL
                    WITH f ORDER BY f.scheduled_departure
                    WITH collect(f) AS flights
                    UNWIND range(0, size(flights)-2) AS i
                    WITH flights[i] AS f1, flights[i+1] AS f2
                    MERGE (f1)-[:NEXT_LEG]->(f2)
                    RETURN count(*) AS cnt
                    """,
                    airline=airline, fnum=fnum,
                )
                cnt = res.single()
                if cnt:
                    created += cnt["cnt"]
        print(f"  {created} relações NEXT_LEG criadas.")

    # ── Clima  →  nó Clima + OBSERVED_AT → Airport ────────────────────────
    def add_clima_to_airport(self, airport_code):
        """Busca clima atual via API e cria nó Clima ligado ao Airport."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:Airport {iata_code: $code})
                RETURN a.latitude AS lat, a.longitude AS lon
                """,
                code=airport_code,
            )
            records = result.data()
            if not records:
                print(f"Airport {airport_code} não encontrado.")
                return

            lat = records[0]["lat"]
            lon = records[0]["lon"]
            clima_raw = get_clima(lat, lon)
            clima = tratar_clima(clima_raw)
            if clima is None:
                print(f"Sem dados de clima para {airport_code}.")
                return

            session.run(
                """
                MATCH (a:Airport {iata_code: $code})
                CREATE (c:Clima {
                    weather: $weather, description: $description,
                    temp: $temp, windspeed: $windspeed,
                    rain: $rain, clouds: $clouds,
                    timestamp: $timestamp
                })
                CREATE (c)-[:OBSERVED_AT]->(a)
                """,
                code=airport_code,
                timestamp=datetime.now(timezone.utc).isoformat(),
                **clima,
            )

    # ── Clima — bulk por aeroporto via Open-Meteo (horário) ──────────────── 
    def create_clima_bulk_for_airport(self, icao_code, lat, lon, start_iso, end_iso):
        """
        Busca clima horário via Open-Meteo para todo o período de uma vez.
        Cria nós Clima com clima_id = '{icao}_{ddMMyyyy}_{HH}' (dedup via MERGE).
        Retorna quantidade de nós criados.
        """
        try:
            hours = get_clima_openmeteo(lat, lon, start_iso, end_iso)
        except Exception as e:
            print(f"    Erro Open-Meteo {icao_code}: {e}")
            return 0

        if not hours:
            return 0

        # Converter date_iso (YYYY-MM-DD) para ddMMyyyy para bater com f.date
        rows = []
        for h in hours:
            parts = h["date_iso"].split("-")   # ['2025','01','01']
            ddmmyyyy = f"{parts[2]}{parts[1]}{parts[0]}"
            hh = h["hour"]  # '00' .. '23'
            rows.append({
                "clima_id": f"{icao_code}_{ddmmyyyy}_{hh}",
                "date": ddmmyyyy,
                "hour": int(hh),
                "timestamp": h["date_iso"] + f"T{hh}:00:00+00:00",
                "weather": h["weather"],
                "description": h["description"],
                "temp": h["temp"],
                "windspeed": h["windspeed"],
                "rain": h["rain"],
                "clouds": h["clouds"],
            })

        # Batch insert com UNWIND — ~9500 rows por aeroporto, dividir em lotes de 2000
        CHUNK = 2000
        total = 0
        for start in range(0, len(rows), CHUNK):
            chunk = rows[start:start + CHUNK]
            with self.driver.session() as session:
                result = session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (a:Airport {icao_code: $icao})
                    MERGE (c:Clima {clima_id: r.clima_id})
                    ON CREATE SET
                        c.weather = r.weather, c.description = r.description,
                        c.temp = r.temp, c.windspeed = r.windspeed,
                        c.rain = r.rain, c.clouds = r.clouds,
                        c.timestamp = r.timestamp, c.date = r.date,
                        c.hour = r.hour
                    MERGE (c)-[:OBSERVED_AT]->(a)
                    RETURN count(c) AS cnt
                    """,
                    rows=chunk, icao=icao_code,
                )
                total += result.single()["cnt"]
        return total

    # ── Liga cada Flight ao Clima do mesmo aeroporto+dia+hora ─────────
    def link_flights_to_clima(self):
        """
        Para cada Flight com ORIGIN, liga ao Clima do mesmo aeroporto+dia+hora
        via HAS_ORIGIN_WEATHER. Usa a hora da partida prevista.
        Processa por data para evitar estouro de memória.
        """
        with self.driver.session() as session:
            dates = session.run(
                "MATCH (f:Flight) RETURN DISTINCT f.date AS dt"
            )
            all_dates = [r["dt"] for r in dates]

        total = 0
        for i, dt in enumerate(sorted(all_dates), 1):
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (f:Flight {date: $dt})-[:ORIGIN]->(a:Airport)
                    WHERE NOT (f)-[:HAS_ORIGIN_WEATHER]->(:Clima)
                      AND f.scheduled_departure IS NOT NULL
                    WITH f, a.icao_code AS icao,
                         substring(f.scheduled_departure, 11, 2) AS hh
                    MATCH (c:Clima {clima_id: icao + '_' + $dt + '_' + hh})
                    MERGE (f)-[:HAS_ORIGIN_WEATHER]->(c)
                    RETURN count(*) AS linked
                    """,
                    dt=dt,
                )
                cnt = result.single()["linked"]
                total += cnt
            if i % 30 == 0:
                print(f"    {i}/{len(all_dates)} datas ({total} voos ligados)")
        print(f"  {total} voos ligados a Clima (ORIGIN).")


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════
def date_range(start, end):
    """Gera datas no formato ddMMyyyy entre start e end (inclusive)."""
    current = datetime.strptime(start, "%d%m%Y")
    end_dt  = datetime.strptime(end,   "%d%m%Y")
    while current <= end_dt:
        yield current.strftime("%d%m%Y")
        current += timedelta(days=1)


def parse_hora(time_str):
    """
    Parseia datetime da API VRA (formato 'dd/MM/yyyy HH:mm') em datetime UTC.
    Retorna None se não for possível parsear.
    """
    if pd.isna(time_str) or not time_str:
        return None
    time_str = str(time_str).strip()
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(time_str, fmt).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
    return None


def calcular_atraso(real_dt, previsto_dt):
    """Retorna atraso em minutos (positivo = atrasou). None se dados faltam."""
    if real_dt is None or previsto_dt is None:
        return None
    delta = (real_dt - previsto_dt).total_seconds() / 60.0
    return round(delta, 1)


def normalize_vra_payload(raw):
    """Converte o retorno da API VRA em DataFrame padronizado."""
    payload = raw

    if isinstance(payload, str):
        payload = payload.strip()
        if not payload or not payload.startswith(("[", "{")):
            return pd.DataFrame()
        payload = json.loads(payload)

    if isinstance(payload, list) and payload and isinstance(payload[0], str):
        payload = json.loads(payload[0])

    if payload is None or payload == []:
        return pd.DataFrame()

    if isinstance(payload, list):
        df = pd.DataFrame(payload)
    elif isinstance(payload, dict):
        records_key = next(
            (k for k, v in payload.items() if isinstance(v, list)), None
        )
        df = pd.DataFrame(payload[records_key]) if records_key else pd.DataFrame([payload])
    else:
        raise TypeError(f"Formato inesperado: {type(payload)}")

    # Padroniza nomes de colunas
    column_aliases = {
        "airline_code":      ["sg_empresa_icao", "empresa_icao"],
        "airline_name":      ["nm_empresa"],
        "origin_code":       ["sg_icao_origem", "sg_iata_origem", "origem"],
        "destination_code":  ["sg_icao_destino", "sg_iata_destino", "destino"],
        "flight_number":     ["nr_voo", "numero_voo"],
        "status":            ["ds_situacao_voo", "situacao_voo"],
        "scheduled_dep":     ["dt_partida_prevista", "hr_partida_prevista"],
        "actual_dep":        ["dt_partida_real", "hr_partida_real"],
        "scheduled_arr":     ["dt_chegada_prevista", "hr_chegada_prevista"],
        "actual_arr":        ["dt_chegada_real", "hr_chegada_real"],
        "equipment_icao":    ["sg_equipamento_icao"],
        "line_type":         ["cd_tipo_linha"],
        "seats":             ["nr_assentos_ofertados"],
        "justification":     ["ds_justificativa"],
    }
    for target, candidates in column_aliases.items():
        if target not in df.columns:
            source = next((c for c in candidates if c in df.columns), None)
            if source:
                df[target] = df[source]

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Main — execução completa (multi-data)
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Configuração ───────────────────────────────────────────────────
    START_DATE = "01012020"   # ddMMyyyy
    END_DATE   = "31122024"
    COLLECT_WEATHER    = True   # clima 1x por aeroporto×dia
    ENRICH_AIRCRAFT_AGE = False  # busca aircraft_age_years via OpenSky
    BATCH_SIZE = 500            # voos por transação UNWIND

    db = Neo4jDB()

    # 0. Garantir constraints/índices (idempotente) ─────────────────────
    # db.clear_all()  # descomente só se quiser limpar tudo
    db.create_constraints()

    # 0b. Carregar credenciais OpenSky (se disponíveis) ──────────────────
    if ENRICH_AIRCRAFT_AGE:
        if load_opensky_credentials("credentials.json"):
            print("OpenSky: credenciais carregadas.")
        else:
            print("OpenSky: credenciais não encontradas — enriquecimento desabilitado.")
            ENRICH_AIRCRAFT_AGE = False

    # 1. Carregar aeroportos do CSV (somente large_airport) ─────────────
    airports = pd.read_csv("data/airports.csv")[
        ["type", "name", "latitude_deg", "longitude_deg", "elevation_ft",
         "iso_country", "iso_region", "municipality",
         "icao_code", "iata_code", "home_link"]
    ]
    airports = airports[airports["type"] == "large_airport"]

    print(f"MERGE {len(airports)} aeroportos (large)...")
    for _, row in airports.iterrows():
        db.create_airport(
            row["type"], row["name"],
            row["latitude_deg"], row["longitude_deg"], row["elevation_ft"],
            row["iso_country"], row["iso_region"], row["municipality"],
            row["icao_code"], row["iata_code"], row["home_link"],
        )
    print("Aeroportos OK.\n")

    # 2. Carregar voos — iterando sobre múltiplas datas ─────────────────
    flight_counter = 0
    airline_equip_seen = set()     # (airline_code, equipment_icao) únicos
    airports_dates_seen = set()    # (icao, date) para clima
    # Para enriquecimento OpenSky: {flight_id -> (callsign, sched_dep_unix)}
    flights_for_opensky: dict[str, tuple[str, int]] = {}
    # Mapa flight_id -> equipment_icao (para criar Aircraft individuais)
    batch_equip_map: dict[str, str] = {}

    # Pré-carregar lookup icao ↔ iata dos aeroportos no CSV
    airport_lookup = {}  # code -> {icao, lat, lon}
    for _, row in airports.iterrows():
        icao = row["icao_code"]
        iata = row["iata_code"]
        info = {"icao": icao, "lat": row["latitude_deg"], "lon": row["longitude_deg"]}
        if pd.notna(icao):
            airport_lookup[icao] = info
        if pd.notna(iata):
            airport_lookup[iata] = info

    for dt in date_range(START_DATE, END_DATE):
        print(f"Buscando voos para {dt}...")
        try:
            vra_raw = get_vra_data(dt)
        except Exception as e:
            print(f"  Erro API VRA {dt}: {e}")
            tm.sleep(2)
            continue

        df = normalize_vra_payload(vra_raw)
        if df.empty:
            print(f"  Nenhum voo retornado para {dt}.")
            continue

        df["date"] = dt
        print(f"  {len(df)} voos encontrados.")

        batch = []
        for idx, row in df.iterrows():
            airline = row.get("airline_code")
            origin  = row.get("origin_code")
            dest    = row.get("destination_code")
            fnum    = row.get("flight_number")
            status  = row.get("status", None)

            if pd.isna(airline) or pd.isna(origin) or pd.isna(dest) or pd.isna(fnum):
                continue

            sched_dep_dt  = parse_hora(row.get("scheduled_dep"))
            actual_dep_dt = parse_hora(row.get("actual_dep"))
            sched_arr_dt  = parse_hora(row.get("scheduled_arr"))
            actual_arr_dt = parse_hora(row.get("actual_arr"))

            delay_dep = calcular_atraso(actual_dep_dt, sched_dep_dt)
            delay_arr = calcular_atraso(actual_arr_dt, sched_arr_dt)

            sched_dep_iso  = sched_dep_dt.isoformat()  if sched_dep_dt  else None
            actual_dep_iso = actual_dep_dt.isoformat() if actual_dep_dt else None
            sched_arr_iso  = sched_arr_dt.isoformat()  if sched_arr_dt  else None
            actual_arr_iso = actual_arr_dt.isoformat() if actual_arr_dt else None

            tail = row.get("equipment_icao", None)
            if pd.notna(tail):
                tail = str(tail).strip()
                airline_equip_seen.add((str(airline).strip(), tail))
            else:
                tail = None

            flight_id = f"{airline}_{fnum}_{dt}_{origin}_{dest}"

            batch.append({
                "flight_id": flight_id, "date": dt,
                "airline_code": airline, "origin_code": origin,
                "destination_code": dest, "flight_number": fnum,
                "status": status,
                "scheduled_dep": sched_dep_iso, "actual_dep": actual_dep_iso,
                "scheduled_arr": sched_arr_iso, "actual_arr": actual_arr_iso,
                "delay_dep_min": delay_dep, "delay_arr_min": delay_arr,
                "equipment_icao": tail,
            })

            # Registra callsign + unix para enriquecimento OpenSky
            if ENRICH_AIRCRAFT_AGE and sched_dep_dt:
                callsign = f"{airline}{fnum}".strip()
                flights_for_opensky[flight_id] = (
                    callsign,
                    int(sched_dep_dt.timestamp()),
                )
                if tail:
                    batch_equip_map[flight_id] = tail

            # Coletar aeroportos×datas para clima (dedup)
            if COLLECT_WEATHER:
                for code in (origin, dest):
                    info = airport_lookup.get(code)
                    if info:
                        airports_dates_seen.add((info["icao"], dt, info["lat"], info["lon"]))

            # Enviar batch quando atingir BATCH_SIZE
            if len(batch) >= BATCH_SIZE:
                db.create_flights_batch(batch)
                flight_counter += len(batch)
                batch = []

        # Enviar restante do dia
        if batch:
            db.create_flights_batch(batch)
            flight_counter += len(batch)
            batch = []

        print(f"  Total acumulado: {flight_counter} voos.")

    print(f"\n{flight_counter} voos criados no total.")

    # 3. Criar nós Aircraft e relações ASSIGNED_TO ──────────────────────
    print(f"Criando {len(airline_equip_seen)} combinações airline×tipo de aeronave...")
    for airline_code, equip in airline_equip_seen:
        db.create_aircraft_and_assign_by_airline(airline_code, equip)
    print("Aeronaves criadas.")

    # 4. Enriquecer voos com Aircraft individuais via OpenSky ────────────
    if ENRICH_AIRCRAFT_AGE and flights_for_opensky:
        print(f"\nEnriquecendo {len(flights_for_opensky):,} voos com Aircraft individuais via OpenSky...")

        # Cache icao24 → metadata para não repetir chamada
        from collections import defaultdict
        icao24_meta_cache: dict[str, dict | None] = {}
        aircraft_updates: list[dict] = []
        enriched = 0
        not_found = 0

        # Agrupa voos em janelas de 2h para minimizar chamadas ao OpenSky
        windows: dict[int, list] = defaultdict(list)
        WINDOW = 7200  # 2 horas em segundos
        for fid, (callsign, dep_unix) in flights_for_opensky.items():
            w = (dep_unix // WINDOW) * WINDOW
            windows[w].append((fid, callsign, dep_unix))

        print(f"  {len(windows)} janelas de 2h a consultar no OpenSky...")
        for w_idx, (w_start, flt_list) in enumerate(sorted(windows.items()), 1):
            w_end = w_start + WINDOW
            opensky_flights = get_opensky_flights_by_time(w_start, w_end)

            # Monta lookup callsign → icao24 para esta janela
            cs_to_hex: dict[str, str] = {}
            for of in opensky_flights:
                cs = (of.get("callsign") or "").strip().upper()
                if cs:
                    cs_to_hex[cs] = of.get("icao24", "")

            # Cruza com os voos VRA desta janela
            for fid, callsign, _ in flt_list:
                hex24 = cs_to_hex.get(callsign.upper())
                if not hex24:
                    not_found += 1
                    continue

                # Busca metadados (com cache)
                if hex24 not in icao24_meta_cache:
                    icao24_meta_cache[hex24] = get_opensky_aircraft_metadata(hex24)

                meta = icao24_meta_cache[hex24]
                if not meta:
                    not_found += 1
                    continue

                registration = meta.get("registration") or ""
                built = meta.get("built")
                age = None
                if built:
                    try:
                        from datetime import datetime as _dt
                        age = float(_dt.now().year - int(str(built)[:4]))
                    except (ValueError, TypeError):
                        pass

                # Busca equipment_icao do voo (está em batch, pegar do dict criado acima)
                equip = batch_equip_map.get(fid, "")

                aircraft_updates.append({
                    "flight_id":       fid,
                    "icao24":          hex24,
                    "registration":    registration,
                    "aircraft_age_years": age,
                    "equipment_icao":  equip,
                })
                enriched += 1

            if w_idx % 50 == 0:
                print(f"  [{w_idx}/{len(windows)}] enriquecidos={enriched}, "
                      f"sem match={not_found}")
            tm.sleep(0.2)  # gentileza com o rate-limit

        # Salva no Neo4j em batch
        CHUNK = 1000
        for start in range(0, len(aircraft_updates), CHUNK):
            db.update_flights_individual_aircraft_batch(aircraft_updates[start:start + CHUNK])

        print(f"  Aircraft individuais criados para {enriched:,} voos "
              f"({not_found:,} sem match no OpenSky).")

    # 5. Criar relações NEXT_LEG (voos consecutivos da mesma rota) ──────
    print("Criando relações NEXT_LEG...")
    db.create_next_legs()
    print("NEXT_LEG criado.")

    # 6. Clima — bulk via Open-Meteo (1 chamada por aeroporto) ────────────
    if COLLECT_WEATHER:
        # Deduplica aeroportos usados como origem
        airport_set = {}
        for (icao, dt, lat, lon) in airports_dates_seen:
            if icao not in airport_set:
                airport_set[icao] = (lat, lon)

        # Usar as datas min/max do dataset para pedir tudo de uma vez
        start_dt = datetime.strptime(START_DATE, "%d%m%Y")
        end_dt   = datetime.strptime(END_DATE,   "%d%m%Y")
        start_iso = start_dt.strftime("%Y-%m-%d")
        end_iso   = end_dt.strftime("%Y-%m-%d")

        print(f"\nColetando clima Open-Meteo para {len(airport_set)} aeroportos ({start_iso} a {end_iso})...")
        clima_total = 0
        for i, (icao, (lat, lon)) in enumerate(sorted(airport_set.items()), 1):
            cnt = db.create_clima_bulk_for_airport(icao, lat, lon, start_iso, end_iso)
            clima_total += cnt
            if i % 20 == 0 or cnt > 0:
                print(f"  {i}/{len(airport_set)} aeroportos ({clima_total} nós Clima criados)")
            tm.sleep(0.3)  # gentileza com Open-Meteo
        print(f"  {clima_total} nós Clima criados no total.")

        # 7. Ligar voos ao Clima do aeroporto de origem no mesmo dia ─────
        print("Ligando voos ao Clima (ORIGIN)...")
        db.link_flights_to_clima()

    db.close()
    print("\nCarga finalizada!")
