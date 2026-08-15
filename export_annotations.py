"""
Person D — hour 0: export annotations.csv for validate.py.

Pulls (episode_id, text, preview_mp4) from the SQL episodes table.
`text` is the concatenation of per-clip `segments[].label` (real language
annotations, e.g. "drop orange bowl, pick up orange bowl" — this is where
the drop/fail/retry weak-label keywords actually live). Falls back to
`task_description` for rows with no segments so the join still has *some*
text to keyword-match against, rather than silently losing the row.

Usage:
  python export_annotations.py                       # all labs, human only
  python export_annotations.py --lab mecka            # one lab
  python export_annotations.py --task fold_clothes    # one task
  python export_annotations.py --limit 5000
"""

import argparse
import sys

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, "/home/eunji/EgoVerse")  # egomimic package lives in the EgoVerse checkout
from egomimic.utils.aws.aws_sql import create_default_engine  # noqa: E402


def segments_to_text(segments) -> str:
    if not segments:
        return ""
    return "; ".join(
        s.get("label") or "" for s in segments if isinstance(s, dict)
    ).strip("; ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default="")
    ap.add_argument("--task", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--human-only", action="store_true", default=True)
    ap.add_argument("--out", default="annotations.csv")
    args = ap.parse_args()

    engine = create_default_engine()

    where = ["zarr_mp4_path IS NOT NULL", "zarr_mp4_path != ''"]
    params = {}
    if args.human_only:
        where.append("embodiment LIKE 'human%'")
    if args.lab:
        where.append("lab = :lab")
        params["lab"] = args.lab
    if args.task:
        where.append("task = :task")
        params["task"] = args.task
    limit_clause = "LIMIT :limit" if args.limit else ""
    if args.limit:
        params["limit"] = args.limit

    query = text(f"""
        SELECT episode_hash, lab, task, embodiment, zarr_mp4_path,
               segments, task_description
        FROM app.episodes
        WHERE {' AND '.join(where)}
        {limit_clause}
    """)
    df = pd.read_sql(query, engine, params=params)
    print(f"pulled {len(df)} rows", file=sys.stderr)

    ann_text = df["segments"].apply(segments_to_text)
    ann_text = ann_text.where(ann_text != "", df["task_description"].fillna(""))

    out = pd.DataFrame({
        "episode_id": df["episode_hash"],
        "text": ann_text,
        "preview_mp4": df["zarr_mp4_path"],
        "lab": df["lab"],
        "task": df["task"],
    })
    out.to_csv(args.out, index=False)
    has_segments = (df["segments"].apply(lambda s: bool(s))).sum()
    print(f"wrote {args.out}: {len(out)} rows, {has_segments} with real segment "
          f"labels ({has_segments/len(out):.1%})", file=sys.stderr)


if __name__ == "__main__":
    main()
