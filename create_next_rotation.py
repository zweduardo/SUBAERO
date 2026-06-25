# create_next_rotation.py — Cria arestas NEXT_ROTATION entre voos consecutivos
#                            do mesmo TIPO de aeronave (equipment_icao) no mesmo dia
#
# Diferença vs NEXT_LEG:
#   NEXT_LEG      → mesmo número de voo (mesma rota recorrente)
#                   Útil para: sazonalidade, padrões de rota
#   NEXT_ROTATION → mesmo equipamento no mesmo dia
#                   Útil para: propagação de atraso (avião chegou tarde → próximo voo atrasa)
#
# Como o grafo atual usa equipment_icao como TIPO (não matrícula individual),
# NEXT_ROTATION conecta voos que usam o mesmo tipo de aeronave no mesmo dia,
# ordenados por horário de partida programada.
#
# Uso:
#   python create_next_rotation.py             → processa todos os equipamentos
#   python create_next_rotation.py --dry-run   → conta sem criar
#   python create_next_rotation.py --since 2026-01-01  → só voos após a data

import argparse
from neo4j import GraphDatabase

NEO4J_URI      = "bolt://192.168.15.118:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "tcc12345"

BATCH_SIZE = 500  # equipamentos processados por transação


class Neo4jDB:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def _run(self, query, **params):
        with self.driver.session() as s:
            return s.run(query, **params).data()

    # ── Conta arestas existentes ──────────────────────────────────────────

    def count_existing(self) -> int:
        result = self._run("MATCH ()-[:NEXT_ROTATION]->() RETURN count(*) AS n")
        return result[0]["n"]

    # ── Lista todos os tipos de equipamento com voos ──────────────────────

    def get_equipment_types(self, since: str | None = None) -> list[str]:
        """
        Retorna chaves de agrupamento distintas: icao24 quando disponível,
        equipment_icao como fallback (COALESCE).
        """
        where = f"WHERE f.scheduled_departure >= '{since}'" if since else ""
        rows = self._run(
            f"""
            MATCH (f:Flight)
            {where}
            WHERE f.scheduled_departure IS NOT NULL
              AND (f.icao24 IS NOT NULL OR f.equipment_icao IS NOT NULL)
            RETURN DISTINCT coalesce(f.icao24, f.airline_code + '_' + f.equipment_icao) AS equip
            ORDER BY equip
            """
        )
        return [r["equip"] for r in rows]

    # ── Cria NEXT_ROTATION para uma chave de equipamento ─────────────────

    def create_rotations_for_type(self, equip: str, since: str | None = None) -> int:
        """
        Para uma dada chave (icao24 individual ou equipment_icao tipo),
        ordena voos por scheduled_departure e cria NEXT_ROTATION entre
        pares consecutivos no mesmo dia (calendário).

        Retorna o número de arestas criadas/mergeadas.
        """
        where_since = f"AND f.scheduled_departure >= '{since}'" if since else ""
        result = self._run(
            f"""
            MATCH (f:Flight)
            WHERE coalesce(f.icao24, f.airline_code + '_' + f.equipment_icao) = $equip
              AND f.scheduled_departure IS NOT NULL
              {where_since}

            // Extrai data do dia (primeiros 10 chars do ISO timestamp)
            WITH f, substring(f.scheduled_departure, 0, 10) AS flight_day
            ORDER BY flight_day, f.scheduled_departure

            // Agrupa por dia
            WITH flight_day, collect(f) AS daily_flights

            // Para cada par consecutivo no mesmo dia, cria a aresta
            UNWIND range(0, size(daily_flights) - 2) AS i
            WITH daily_flights[i] AS f1, daily_flights[i+1] AS f2
            MERGE (f1)-[:NEXT_ROTATION]->(f2)
            RETURN count(*) AS created
            """,
            equip=equip,
        )
        return result[0]["created"] if result else 0

    # ── Conta quantas arestas seriam criadas (sem criar) ─────────────────

    def count_rotations_for_type(self, equip: str) -> int:
        result = self._run(
            """
            MATCH (f:Flight)
            WHERE coalesce(f.icao24, f.airline_code + '_' + f.equipment_icao) = $equip
              AND f.scheduled_departure IS NOT NULL
            WITH f, substring(f.scheduled_departure, 0, 10) AS flight_day
            ORDER BY flight_day, f.scheduled_departure
            WITH flight_day, collect(f) AS daily_flights
            RETURN sum(size(daily_flights) - 1) AS pairs
            """,
            equip=equip,
        )
        return result[0]["pairs"] if result else 0

    # ── Índice (melhora performance das queries acima) ────────────────────

    def ensure_index(self):
        with self.driver.session() as s:
            try:
                s.run(
                    "CREATE INDEX IF NOT EXISTS FOR (f:Flight) "
                    "ON (f.equipment_icao, f.scheduled_departure)"
                )
            except Exception:
                pass  # já existe


def main():
    parser = argparse.ArgumentParser(
        description="Cria arestas NEXT_ROTATION entre voos consecutivos do mesmo equipamento"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Conta pares sem criar arestas")
    parser.add_argument("--since", default=None,
                        help="Processa apenas voos a partir dessa data (YYYY-MM-DD)")
    args = parser.parse_args()

    db = Neo4jDB()
    try:
        # Garante índice composto
        print("[0/3] Verificando índice composto (equipment_icao, scheduled_departure)...")
        db.ensure_index()

        existing = db.count_existing()
        print(f"  Arestas NEXT_ROTATION existentes: {existing:,}")

        print("\n[1/3] Buscando tipos de equipamento...")
        types = db.get_equipment_types(since=args.since)
        print(f"  {len(types)} tipos com voos{' após ' + args.since if args.since else ''}")

        if args.dry_run:
            print("\n[dry-run] Contando pares que seriam criados...")
            total_pairs = 0
            for equip in types:
                n = db.count_rotations_for_type(equip)
                total_pairs += n
                print(f"  {equip:8s}  ->{n:,} pares")
            print(f"\n  Total estimado: {total_pairs:,} arestas NEXT_ROTATION")
            return

        print("\n[2/3] Criando arestas NEXT_ROTATION...")
        total_created = 0
        for i, equip in enumerate(types, 1):
            n = db.create_rotations_for_type(equip, since=args.since)
            total_created += n
            print(f"  [{i:3d}/{len(types)}] {equip:8s}  ->{n:,} arestas")

        print(f"\n[3/3] Concluído.")
        print(f"  Arestas NEXT_ROTATION criadas: {total_created:,}")
        print(f"  Total agora no banco          : {db.count_existing():,}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
