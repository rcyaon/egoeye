"""
Small local (non-Modal) audit run so the search demo page has real quality
scores instead of "unaudited" placeholders. Same load path as audit_modal.py's
load_wrist(), just sequential instead of fanned out — fine for a few hundred
episodes, not the full corpus.

Usage:
  python local_audit.py --task cup_on_saucer --lab mecka --limit 200 --out audit_results.parquet
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/eunji/EgoVerse")
from egomimic.utils.aws.aws_sql import create_default_engine
from eyekit import score_episode
from make_demo_figs import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="")
    ap.add_argument("--lab", default="")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default="audit_results.parquet")
    args = ap.parse_args()

    engine = create_default_engine()
    where = ["embodiment LIKE 'human%'", "zarr_processed_path IS NOT NULL",
             "zarr_processed_path != ''"]
    params = {}
    if args.task:
        where.append("task = :task"); params["task"] = args.task
    if args.lab:
        where.append("lab = :lab"); params["lab"] = args.lab
    from sqlalchemy import text
    q = text(f"""SELECT episode_hash AS episode_id, zarr_processed_path, task, lab
                 FROM app.episodes WHERE {' AND '.join(where)} LIMIT :limit""")
    params["limit"] = args.limit
    df = pd.read_sql(q, engine, params=params)
    print(f"auditing {len(df)} episodes", file=sys.stderr)

    rows, errors = [], 0
    for i, row in df.iterrows():
        try:
            xyz = load(row["zarr_processed_path"], None, 0)
            rep = score_episode(str(row["episode_id"]), xyz, fps=args.fps)
            rows.append(rep.to_dict())
        except Exception as e:
            errors += 1
            rows.append({"episode_id": str(row["episode_id"]), "error": repr(e)})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(df)} ({errors} errors)", file=sys.stderr)

    out = pd.DataFrame(rows)
    if "error" not in out:
        out["error"] = ""
    out["error"] = out["error"].fillna("")
    out.to_parquet(args.out)
    ok = out[out["error"] == ""]
    print(f"wrote {args.out}: {len(ok)}/{len(out)} succeeded", file=sys.stderr)
    if len(ok):
        print(f"prevalence: {ok['failure_flag'].astype(bool).mean():.1%}", file=sys.stderr)


if __name__ == "__main__":
    main()
