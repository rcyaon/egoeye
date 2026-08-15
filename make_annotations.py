"""
make_annotations.py — export annotations.csv for validate.py (Person D helper).

Pulls the weak-label text + preview video path straight from the SQL episode
table (no zarr needed): episode_id, text, preview_mp4. `text` is the episode's
task_description — the keyword matcher in validate.py scans it for drop/fail/
retry/etc. `preview_mp4` is the clip you watch for precision@10.

    python make_annotations.py                 # all human_bimanual episodes
    python make_annotations.py --meta episodes_audit_strat.csv   # just the audited set

Requires the egomimic repo on PYTHONPATH + ~/.egoverse_env (same as sync_s3).
"""
from __future__ import annotations
import argparse
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=None,
                    help="optional episodes csv to scope to (episode_id column)")
    ap.add_argument("--out", default="annotations.csv")
    args = ap.parse_args()

    from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df
    df = episode_table_to_df(create_default_engine())
    df = df[df["is_deleted"] == False]                       # noqa: E712

    out = pd.DataFrame({
        "episode_id": df["episode_hash"].astype(str),
        # text the weak-label matcher scans; join task name in case a description
        # is blank, so wash_dishes/fold_clothes still contribute a token.
        "text": (df["task_description"].fillna("").astype(str) + " "
                 + df["task"].fillna("").astype(str)).str.strip(),
        "preview_mp4": df["zarr_mp4_path"].astype(str),
    })

    if args.meta:
        ids = pd.read_csv(args.meta, dtype={"episode_id": str})["episode_id"].astype(str)
        out = out[out["episode_id"].isin(set(ids))]

    out.to_csv(args.out, index=False)
    nonempty = (out["text"].str.len() > 0).sum()
    print(f"wrote {args.out}: {len(out)} rows | {nonempty} with text | "
          f"{(out['preview_mp4'].str.len() > 0).sum()} with preview_mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
