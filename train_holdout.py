"""
train_holdout.py
----------------
Airport holdout experiment: treina HGT excluindo um conjunto de aeroportos
completamente (treino + val + test), depois avalia nesses aeroportos retidos.

Isso testa generalização espacial real — o modelo nunca viu dados desses aeroportos.

Usage:
    python train_holdout.py
    python train_holdout.py --holdout MAO,FOR,POA,GYN,VCP
    python train_holdout.py --holdout MAO,FOR,POA,GYN,VCP --epochs 50 --output model_holdout.pt
    python train_holdout.py --eval-only --holdout MAO,FOR,POA,GYN,VCP --output model_holdout.pt
"""

import argparse
import csv
import json
import random

import numpy as np
import torch
import torch.nn as nn

from build_train_iterative_rotation import (
    Neo4jLoader,
    build_period_graph,
    FlightHGT,
    build_model,
    train_step,
    evaluate,
    split_periods,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
)

# Default airports to hold out — one from each major region of Brazil
DEFAULT_HOLDOUT = ["MAO", "FOR", "POA", "GYN", "VCP"]


def parse_args():
    p = argparse.ArgumentParser(description="Airport holdout generalization test")
    p.add_argument("--holdout",   default=",".join(DEFAULT_HOLDOUT),
                   help="Comma-separated IATA codes to hold out entirely from training")
    p.add_argument("--epochs",    default=50,    type=int)
    p.add_argument("--lr",        default=1e-3,  type=float)
    p.add_argument("--wd",        default=1e-4,  type=float)
    p.add_argument("--hidden",    default=64,    type=int)
    p.add_argument("--heads",     default=4,     type=int)
    p.add_argument("--dropout",   default=0.3,   type=float)
    p.add_argument("--seed",      default=42,    type=int)
    p.add_argument("--log-every", default=5,     type=int)
    p.add_argument("--output",    default="model_holdout_hgt.pt")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training — load existing model and only evaluate on holdout airports")
    return p.parse_args()


def main(args):
    holdout_airports = [a.strip().upper() for a in args.holdout.split(",") if a.strip()]
    print(f"[holdout] airports excluded from training: {holdout_airports}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        all_airports = loader.get_large_airports()
        train_airports = [a for a in all_airports if a not in holdout_airports]
        missing = [a for a in holdout_airports if a not in all_airports]
        if missing:
            print(f"[warn] airports not found in Neo4j: {missing}")

        print(f"[data] total large airports: {len(all_airports)}")
        print(f"[data] training airports: {len(train_airports)}")
        print(f"[data] holdout airports:  {len(holdout_airports)}")

        # --- Periods for training set (excludes holdout airports) ---
        all_periods  = loader.get_available_periods(train_airports)
        train_periods, val_periods, test_periods = split_periods(all_periods, seed=args.seed)
        print(f"[split] train={len(train_periods)} val={len(val_periods)} test={len(test_periods)}")

        # Save split for reference
        with open("periods_split_holdout.json", "w") as f:
            json.dump({
                "holdout_airports": holdout_airports,
                "train": [list(p) for p in train_periods],
                "val":   [list(p) for p in val_periods],
                "test":  [list(p) for p in test_periods],
            }, f, indent=2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.MSELoss()

        if not args.eval_only:
            # --- Build a sample graph to initialize model ---
            model = None
            for period in train_periods:
                sample_graph, _, _ = build_period_graph(loader, *period)
                if sample_graph is not None:
                    # Force HGT regardless of args
                    class _FakeArgs:
                        model   = "hgt"
                        hidden  = args.hidden
                        heads   = args.heads
                        dropout = args.dropout
                    model = build_model(_FakeArgs(), sample_graph)
                    del sample_graph
                    break
            if model is None:
                raise RuntimeError("No valid training period found to initialize model.")

            model = model.to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[model] HGT — {n_params:,} params — device={device}\n")

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )

            # --- Cache graphs ---
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

            # --- Training loop ---
            best_val_mae = float("inf")
            best_state   = None
            metrics      = []

            for epoch in range(1, args.epochs + 1):
                idx_order = list(range(len(train_graphs)))
                random.shuffle(idx_order)
                epoch_losses = []
                for i in idx_order:
                    loss = train_step(model, train_graphs[i], optimizer, criterion, device)
                    epoch_losses.append(loss)

                avg_train = float(np.mean(epoch_losses)) if epoch_losses else 0.0

                val_maes = [evaluate(model, g, criterion, device)[1] for g in val_graphs]
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

            # --- In-distribution test ---
            model.load_state_dict(best_state)
            test_maes = [evaluate(model, g, criterion, device)[1] for g in test_graphs]
            avg_test_mae = float(np.mean(test_maes)) if test_maes else 0.0
            print(f"\n[in-distribution] Test MAE (seen airports, unseen months): {avg_test_mae:.1f} min")
            del train_graphs, val_graphs, test_graphs

            # Save model and metrics
            torch.save(best_state, args.output)
            print(f"[model] saved to '{args.output}'")

            with open("metrics_holdout.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_mse", "val_mae"])
                writer.writerows(metrics)
            print("[metrics] saved to 'metrics_holdout.csv'")

        else:
            # eval-only: load existing model
            print(f"[eval-only] loading model from '{args.output}'")
            # Need a sample graph to infer architecture
            all_holdout_periods = loader.get_available_periods(holdout_airports)
            sample_graph = None
            for period in all_holdout_periods:
                g, _, _ = build_period_graph(loader, *period)
                if g is not None:
                    sample_graph = g
                    break
            if sample_graph is None:
                raise RuntimeError("No valid holdout period found to initialize model architecture.")

            class _FakeArgs:
                model   = "hgt"
                hidden  = args.hidden
                heads   = args.heads
                dropout = args.dropout
            model = build_model(_FakeArgs(), sample_graph)
            del sample_graph

            state = torch.load(args.output, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model = model.to(device)
            print(f"[model] loaded from '{args.output}'")
            best_state = state  # for eval below

        # ----------------------------------------------------------------
        # --- Out-of-distribution evaluation on holdout airports ---
        # ----------------------------------------------------------------
        print(f"\n[holdout eval] building graphs for held-out airports: {holdout_airports}")
        holdout_periods = loader.get_available_periods(holdout_airports)
        print(f"[holdout eval] {len(holdout_periods)} periods available")

        holdout_results = []
        skipped = 0
        for period in holdout_periods:
            g, _, _ = build_period_graph(loader, *period)
            if g is None:
                skipped += 1
                continue
            iata, year, month = period
            n_flights = g["flight"].x.shape[0]
            _, mae = evaluate(model, g, criterion, device)
            holdout_results.append({
                "iata": iata, "year": year, "month": month,
                "n_flights": n_flights, "mae_min": mae,
            })
            print(f"  {iata} {year}-{month:02d}: MAE={mae:.1f} min  ({n_flights} flights)")

        if skipped:
            print(f"  ({skipped} periods skipped — fewer than 10 flights)")

        if not holdout_results:
            print("[warn] No valid holdout periods found.")
            return

        # Summary by airport
        print("\n[holdout summary by airport]")
        by_iata = {}
        for r in holdout_results:
            by_iata.setdefault(r["iata"], []).append(r["mae_min"])
        for iata in sorted(by_iata):
            maes = by_iata[iata]
            print(f"  {iata}: MAE={np.mean(maes):.1f} min  ({len(maes)} periods, "
                  f"min={min(maes):.1f}, max={max(maes):.1f})")

        overall_mae = float(np.mean([r["mae_min"] for r in holdout_results]))
        print(f"\n[result] MAE on holdout airports (never seen): {overall_mae:.1f} min")
        if not args.eval_only:
            print(f"[reference] MAE on seen airports (test split): {avg_test_mae:.1f} min")

        # Save holdout results
        out_csv = args.output.replace(".pt", "_holdout_results.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iata","year","month","n_flights","mae_min"])
            writer.writeheader()
            writer.writerows(holdout_results)
        print(f"[output] holdout results saved to '{out_csv}'")

    finally:
        loader.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
