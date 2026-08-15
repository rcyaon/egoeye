"""
Two checks:
  A) Weak labels from annotation text (drop/fail/retry keywords) -> agreement
     table. Pitch: physics-based detection VALIDATED against text, not
     dependent on it.
  B) precision@10: prints preview MP4 paths for your 10 most-confident
     failure calls. Watch them. Count hits. NEVER skip this.

Usage:
  python validate.py audit_results.parquet annotations.csv
annotations.csv: episode_id, text[, preview_mp4]  (export via sql_tutorial.ipynb)
"""

import sys, re
import pandas as pd

FAIL_WORDS = re.compile(
    r"\b(drop|dropped|drops|fail|failed|failure|retry|retries|redo|"
    r"mistake|fumble|slip|slipped|missed|wrong|again)\b", re.I)

def main(audit_path: str, ann_path: str):
    audit = pd.read_parquet(audit_path)
    if "error" in audit:
        audit = audit[audit["error"] == ""]
    ann = pd.read_csv(ann_path, dtype={"episode_id": str})
    ann["weak_fail"] = ann["text"].fillna("").str.contains(FAIL_WORDS)
    df = audit.merge(ann, on="episode_id", how="inner")
    print(f"matched {len(df)} episodes\n")

    # A) agreement with weak labels
    ct = pd.crosstab(df["failure_flag"], df["weak_fail"],
                     rownames=["ours"], colnames=["text says fail"])
    print("== agreement with annotation keywords ==")
    print(ct, "\n")
    both = int(((df.failure_flag) & (df.weak_fail)).sum())
    ours = int(df.failure_flag.sum()); text = int(df.weak_fail.sum())
    if ours and text:
        print(f"precision vs weak labels: {both/ours:.2f}   "
              f"recall vs weak labels: {both/text:.2f}")
        print("(weak labels are noisy — treat as sanity check, not truth)\n")

    # B) precision@10 gallery — go watch these
    top = df.sort_values("failure_score", ascending=False).head(10)
    print("== WATCH THESE: top-10 most-confident failure calls ==")
    cols = ["episode_id", "failure_score", "n_impulses", "rf_small_ratio",
            "eye_opening"]
    if "preview_mp4" in top:
        cols.append("preview_mp4")
    print(top[cols].to_string(index=False))
    print("\ncount how many are real failures -> that's precision@10 for the slide.")

if __name__ == "__main__":
    main(*sys.argv[1:3])
