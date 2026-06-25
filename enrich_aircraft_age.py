# enrich_aircraft_age.py — Reestrutura Aircraft nodes por (airline × tipo) e enriquece
#
# O que faz:
#   1. MIGRAÇÃO: recria Aircraft nodes por (airline_code × equipment_icao),
#      redireciona ASSIGNED_TO e remove os antigos nodes só por tipo.
#   2. ANAC RAB: calcula idade média real por tipo ICAO (frota ativa no Brasil)
#   3. Fallback estático: ano de introdução por tipo (Jane's / Wikipedia)
#   4. Grava generation_age_years + is_low_cost + node_key em cada Aircraft node
#
# Uso:
#   python enrich_aircraft_age.py            → migra e atualiza Neo4j
#   python enrich_aircraft_age.py --dry-run  → só imprime, não grava

import argparse
import io
import requests
import pandas as pd
from datetime import datetime
from neo4j import GraphDatabase

# ═══════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════
NEO4J_URI      = "bolt://192.168.15.118:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "tcc12345"

CURRENT_YEAR = datetime.now().year

RAB_URL = "https://sistemas.anac.gov.br/dadosabertos/Aeronaves/RAB/dados_aeronaves.csv"

# Companhias low-cost (código ICAO VRA)
LOW_COST_AIRLINES = {
    "GLO",   # GOL
    "AZU",   # Azul
    "ONE",   # MAP Linhas Aéreas
    "PTB",   # Passaredo / Voepass
    "CGH",   # Voepass
    "TWO",   # Two Flex
    "JJE",   # JetSMART Brasil
    "OKJ",   # Sky Airline
    "MOH",   # Mooah
    "VRG",   # Varig (histórico, GOL)
}

# Fallback: ano de início de produção por tipo ICAO
TYPE_INTRO_YEAR: dict[str, int] = {
    "B735": 1988, "B737": 1968, "B738": 1998, "B739": 2000,
    "B38M": 2017, "B39M": 2019,
    "B752": 1983, "B753": 1985,
    "B762": 1981, "B763": 1982, "B764": 1997,
    "B772": 1995, "B77W": 2004, "B77L": 2004,
    "B788": 2011, "B789": 2013, "B78X": 2017,
    "A306": 1974, "A310": 1983,
    "A318": 2003, "A319": 1995, "A320": 1988, "A321": 1994,
    "A19N": 2016, "A20N": 2016, "A21N": 2017,
    "A332": 1994, "A333": 1994,
    "A342": 1991, "A343": 1993, "A345": 2002, "A346": 2004,
    "A359": 2014, "A35K": 2018, "A388": 2007,
    "E170": 2004, "E175": 2005, "E190": 2004, "E195": 2006,
    "E7W5": 2018, "E290": 2018, "E295": 2021,
    "ERJ": 1989, "E135": 1999, "E140": 2001, "E145": 1996,
    "AT45": 1984, "AT72": 1989, "AT76": 1997,
    "CRJ2": 1992, "CRJ7": 2001, "CRJ9": 2001, "CRJX": 2007,
    "DH8A": 1984, "DH8B": 1989, "DH8C": 1994, "DH8D": 2000,
    "C208": 1984, "C172": 1956, "C152": 1977,
    "BE20": 1974, "BE9L": 1979, "PC12": 1991, "SF34": 1983,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Migração: Aircraft nodes por (airline × tipo)
# ═══════════════════════════════════════════════════════════════════════════

def migrate_to_airline_aircraft(driver, dry_run: bool = False):
    """
    Reestrutura os nós Aircraft de per-tipo para per-(airline × tipo).

    Antes: Aircraft {equipment_icao: "A320"}  ← todos os voos A320 de todas airlines
    Depois: Aircraft {node_key: "GLO_A320", airline_code: "GLO", equipment_icao: "A320"}
            Aircraft {node_key: "AZU_A320", airline_code: "AZU", equipment_icao: "A320"}
    """
    with driver.session() as s:
        pairs = s.run(
            """
            MATCH (f:Flight)
            WHERE f.airline_code IS NOT NULL AND f.equipment_icao IS NOT NULL
            RETURN DISTINCT f.airline_code AS airline, f.equipment_icao AS equip
            ORDER BY airline, equip
            """
        ).data()

    print(f"  {len(pairs)} combinações (airline × tipo) encontradas.")

    if dry_run:
        for p in pairs[:20]:
            print(f"    {p['airline']}_{p['equip']}")
        if len(pairs) > 20:
            print(f"    ... e mais {len(pairs)-20}")
        return

    # 1. Cria novos Aircraft nodes por (airline × tipo)
    rows = [{"nk": f"{p['airline']}_{p['equip']}", "airline": p["airline"], "equip": p["equip"]}
            for p in pairs]
    with driver.session() as s:
        s.run(
            """
            UNWIND $rows AS r
            MERGE (ac:Aircraft {node_key: r.nk})
            ON CREATE SET ac.airline_code = r.airline,
                          ac.equipment_icao = r.equip,
                          ac.node_key = r.nk
            """,
            rows=rows,
        )
    print(f"  {len(rows)} Aircraft nodes criados/mergeados.")

    # 2. Redireciona ASSIGNED_TO (por airline para evitar timeout)
    airlines = list({p["airline"] for p in pairs})
    total_redirected = 0
    for airline in airlines:
        with driver.session() as s:
            result = s.run(
                """
                MATCH (f:Flight {airline_code: $airline})
                WHERE f.equipment_icao IS NOT NULL
                WITH f, $airline + '_' + f.equipment_icao AS nk
                MATCH (new_ac:Aircraft {node_key: nk})
                OPTIONAL MATCH (f)-[old:ASSIGNED_TO]->(old_ac:Aircraft)
                WHERE old_ac.airline_code IS NULL
                DELETE old
                MERGE (f)-[:ASSIGNED_TO]->(new_ac)
                RETURN count(f) AS updated
                """,
                airline=airline,
            ).single()
            total_redirected += result["updated"] if result else 0
    print(f"  {total_redirected} voos redirecionados para Aircraft individuais.")

    # 3. Remove nodes tipo-only orphaned
    with driver.session() as s:
        result = s.run(
            """
            MATCH (ac:Aircraft)
            WHERE ac.airline_code IS NULL
            DETACH DELETE ac
            RETURN count(ac) AS deleted
            """
        ).single()
        print(f"  {result['deleted'] if result else 0} Aircraft tipo-only removidos.")


# ═══════════════════════════════════════════════════════════════════════════
#  Camada 1: ANAC RAB
# ═══════════════════════════════════════════════════════════════════════════

def fetch_rab_age_by_type(timeout: int = 60) -> dict[str, float]:
    """
    Baixa o RAB da ANAC e retorna dict {equipment_icao → idade_média_anos}
    calculada sobre a frota ativa registrada no Brasil.
    """
    print(f"  Baixando RAB da ANAC...")
    try:
        resp = requests.get(RAB_URL, timeout=timeout)
        resp.raise_for_status()
        raw = resp.content

        # Tenta diferentes encodings e skiprows para lidar com BOM / linha de metadata
        df = None
        for enc in ("utf-8-sig", "latin-1", "utf-16"):
            for skip in (0, 1, 2):
                try:
                    candidate = pd.read_csv(
                        io.StringIO(raw.decode(enc, errors="replace")),
                        sep=";", dtype=str, low_memory=False,
                        skiprows=skip,
                    )
                    # Considera válido se tiver pelo menos 5 colunas e mais de 100 linhas
                    if candidate.shape[1] >= 5 and len(candidate) > 100:
                        df = candidate
                        break
                except Exception:
                    continue
            if df is not None:
                break

        if df is None:
            print("    AVISO: não foi possível parsear o RAB.")
            return {}

        print(f"    {len(df)} aeronaves no RAB.")
    except Exception as exc:
        print(f"    AVISO: falha ao baixar RAB — {exc}")
        return {}

    df.columns = [c.strip().upper() for c in df.columns]

    col_type   = next((c for c in df.columns if "ICAO" in c and "TIPO" in c), None)
    col_year   = next((c for c in df.columns if "ANO"  in c and "FABRIC" in c), None)
    col_cancel = next((c for c in df.columns if "CANCEL" in c), None)

    if not col_type or not col_year:
        print(f"    AVISO: colunas não encontradas ({list(df.columns)[:10]})")
        return {}

    if col_cancel:
        df = df[df[col_cancel].isna() | (df[col_cancel].str.strip() == "")]

    df[col_year] = pd.to_numeric(df[col_year], errors="coerce")
    df = df.dropna(subset=[col_year, col_type])
    df = df[df[col_year] > 1940]
    df["age"] = CURRENT_YEAR - df[col_year]
    df[col_type] = df[col_type].str.strip().str.upper()

    age_map = df.groupby(col_type)["age"].mean().round(1).to_dict()
    print(f"    {len(age_map)} tipos com dados de idade no RAB.")
    return age_map


# ═══════════════════════════════════════════════════════════════════════════
#  Camada 2: fallback estático
# ═══════════════════════════════════════════════════════════════════════════

def static_age(equip: str) -> float | None:
    intro = TYPE_INTRO_YEAR.get(equip.upper())
    if intro is None:
        return None
    return float(CURRENT_YEAR - intro + 3)


# ═══════════════════════════════════════════════════════════════════════════
#  Neo4j: busca + gravação
# ═══════════════════════════════════════════════════════════════════════════

def get_aircraft_airline_types(driver) -> list[dict]:
    """Retorna lista de {node_key, airline_code, equipment_icao} para todos Aircraft."""
    with driver.session() as s:
        return s.run(
            """
            MATCH (ac:Aircraft)
            WHERE ac.node_key IS NOT NULL
            RETURN ac.node_key AS node_key,
                   ac.airline_code AS airline_code,
                   ac.equipment_icao AS equipment_icao
            """
        ).data()


def update_aircraft_ages(driver, aircraft_list: list[dict],
                         age_map: dict[str, float], dry_run: bool = False):
    """
    Grava em cada Aircraft node:
      - generation_age_years  (por tipo, do RAB ou fallback)
      - is_low_cost           (0 ou 1, por airline_code)
    """
    rows = []
    not_found = []
    for ac in aircraft_list:
        equip   = (ac["equipment_icao"] or "").upper()
        airline = ac["airline_code"] or ""
        nk      = ac["node_key"]

        age = age_map.get(equip)
        if age is None:
            not_found.append(equip)

        rows.append({
            "node_key":    nk,
            "age":         age,
            "is_low_cost": 1.0 if airline in LOW_COST_AIRLINES else 0.0,
        })

    # Fallback global para tipos sem dados
    known_ages = [r["age"] for r in rows if r["age"] is not None]
    fallback   = round(sum(known_ages) / len(known_ages), 1) if known_ages else 15.0
    for r in rows:
        if r["age"] is None:
            r["age"] = fallback

    if dry_run:
        print("  [dry-run] Amostra dos valores que seriam gravados:")
        for r in sorted(rows, key=lambda x: x["node_key"])[:20]:
            print(f"    {r['node_key']:20s}  age={r['age']:5.1f}  low_cost={int(r['is_low_cost'])}")
        if len(rows) > 20:
            print(f"    ... e mais {len(rows)-20}")
        if not_found:
            tipos_uniq = sorted(set(not_found))
            print(f"  AVISO: {len(tipos_uniq)} tipo(s) sem RAB/fallback → usaram media {fallback}a: {tipos_uniq}")
        return

    with driver.session() as s:
        s.run(
            """
            UNWIND $rows AS r
            MATCH (ac:Aircraft {node_key: r.node_key})
            SET ac.generation_age_years = r.age,
                ac.is_low_cost          = r.is_low_cost
            """,
            rows=rows,
        )
    print(f"  {len(rows)} Aircraft nodes atualizados.")
    if not_found:
        print(f"  AVISO: {len(set(not_found))} tipo(s) sem RAB usaram fallback de {fallback}a.")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Migra Aircraft para airline×tipo e enriquece com age + is_low_cost"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Imprime valores sem gravar no Neo4j")
    parser.add_argument("--skip-migration", action="store_true",
                        help="Pula a migração (já feita anteriormente)")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        if not args.skip_migration:
            print("[0/3] Migrando Aircraft nodes para (airline × tipo)...")
            migrate_to_airline_aircraft(driver, dry_run=args.dry_run)

        print("\n[1/3] Buscando Aircraft nodes no Neo4j...")
        aircraft_list = get_aircraft_airline_types(driver)
        print(f"  {len(aircraft_list)} Aircraft nodes encontrados.")

        print("\n[2/3] Buscando idades médias no ANAC RAB...")
        rab_ages = fetch_rab_age_by_type()

        # Monta age_map: RAB tem prioridade, fallback estático para o resto
        age_map: dict[str, float] = {}
        all_types = {(ac["equipment_icao"] or "").upper() for ac in aircraft_list}
        for equip in all_types:
            if equip in rab_ages:
                age_map[equip] = rab_ages[equip]
            else:
                age = static_age(equip)
                if age is not None:
                    age_map[equip] = age

        print(f"  {len(age_map)}/{len(all_types)} tipos com dados (RAB + fallback estático).")

        print("\n[3/3] Gravando no Neo4j...")
        update_aircraft_ages(driver, aircraft_list, age_map, dry_run=args.dry_run)

        # Resumo
        ages = list(age_map.values())
        low_cost = sum(1 for ac in aircraft_list if ac.get("airline_code") in LOW_COST_AIRLINES)
        print(f"\nResumo:")
        print(f"  Aircraft nodes     : {len(aircraft_list)}")
        print(f"  Low-cost           : {low_cost}")
        print(f"  Legacy/cargo       : {len(aircraft_list) - low_cost}")
        if ages:
            print(f"  Idade mín (tipo)   : {min(ages):.1f} anos")
            print(f"  Idade máx (tipo)   : {max(ages):.1f} anos")
            print(f"  Idade média (tipo) : {sum(ages)/len(ages):.1f} anos")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
