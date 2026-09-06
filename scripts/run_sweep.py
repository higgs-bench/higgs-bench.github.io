"""Full sweep: all versions x models x techniques x seeds.

Resumable - finished cells are cached as JSON and skipped.
Parallel across cells (3 processes x 2 threads = 6 cores).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from higgs_bench.data import load_config
from higgs_bench.models import MODEL_NAMES, SUPPORTS_FOCAL
from higgs_bench.runner import run_cell, run_key
from higgs_bench.techniques import TECHNIQUES, valid_cells

SEEDS = [0, 1, 2, 3, 4]
VERSIONS = ["A", "B", "C"]


def build_jobs(versions, seeds, models, techniques):
    cells = valid_cells(models, techniques, SUPPORTS_FOCAL)
    jobs = []
    for v in versions:
        for (m, t) in cells:
            for s in seeds:
                jobs.append((v, m, t, s))
    # slowest models first so they aren't left stranded at the end
    order = {"adaboost": 0, "voting_4": 1, "voting_3": 2}
    jobs.sort(key=lambda j: order.get(j[1], 9))
    return jobs


def _work(payload):
    job, force, save_preds = payload
    v, m, t, s = job
    cfg = load_config()
    row = run_cell(v, m, t, s, cfg, force=force, save_predictions=save_preds)
    return row["key"], row["status"], row.get("error", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--versions", nargs="+", default=VERSIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--techniques", nargs="+", default=TECHNIQUES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    runs_dir = Path(cfg["paths"]["results_dir"]) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.versions, args.seeds, args.models, args.techniques)
    if args.save_predictions:
        done = {p.stem for p in (Path(cfg["paths"]["results_dir"]) / "preds").glob("*.npz")}
    else:
        done = {p.stem for p in runs_dir.glob("*.json")}
    todo = jobs if args.force else [j for j in jobs if run_key(*j) not in done]

    print(f"total cells : {len(jobs)}")
    print(f"already done: {len(jobs) - len(todo)}")
    print(f"to run      : {len(todo)}")
    print(f"workers     : {args.workers}")
    if args.dry_run:
        for j in todo[:20]:
            print("  ", run_key(*j))
        print("  ..." if len(todo) > 20 else "")
        return

    if not todo:
        print("nothing to do")
        return

    t0 = time.time()
    ok = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_work, (j, args.force, args.save_predictions)): j
                for j in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            key, status, err = fut.result()
            if status == "ok":
                ok += 1
            else:
                failed += 1
                print(f"  !! {key}: {err}")
            el = time.time() - t0
            rate = el / i
            print(f"[{i}/{len(todo)}] ok={ok} failed={failed} "
                  f"elapsed={el/60:.1f}m eta={(len(todo)-i)*rate/60:.1f}m",
                  flush=True)

    print(f"\nSWEEP DONE  ok={ok} failed={failed} "
          f"wall={(time.time()-t0)/60:.1f} min")
    if failed:
        print("inspect failures: results\\runs\\*.json where status=failed")
        sys.exit(1)


if __name__ == "__main__":
    main()