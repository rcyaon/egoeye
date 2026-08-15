"""Validate an audit run before anyone builds a slide on top of it.

    python check_audit.py audit_results.parquet

Exits non-zero if a HARD check fails. Run it after the smoke test, after the
calibration run, and after the full run — the failure modes are different at
each scale and every one of them is silent in the parquet.
"""

import sys
import numpy as np
import pandas as pd

HARD, SOFT = [], []


def hard(ok, label, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}  {detail}".rstrip())
    if not ok:
        HARD.append(label)


def soft(ok, label, detail=""):
    print(f"{'PASS' if ok else 'WARN'}  {label}  {detail}".rstrip())
    if not ok:
        SOFT.append(label)


def main(path="audit_results.parquet"):
    res = pd.read_parquet(path)
    n = len(res)
    print(f"== {path}: {n} rows, {len(res.columns)} columns ==\n")

    # ---- 1. did it run at all -------------------------------------------
    hard(n > 0, "non-empty", f"{n} rows")
    err = res["error"].fillna("") if "error" in res else pd.Series([""] * n)
    ok = res[err == ""]
    rate = len(ok) / n if n else 0
    hard(rate > 0, "some episodes succeeded", f"{len(ok)}/{n}")
    soft(rate >= 0.95, "error rate under 5%", f"{1-rate:.1%} failed")
    if (err != "").any():
        print("\n  most common errors:")
        print("   ", err[err != ""].str.slice(0, 100).value_counts().head(5)
              .to_string().replace("\n", "\n    "), "\n")

    if not len(ok):
        return finish()

    # ---- 2. THE correctness check: chunk padding ------------------------
    # zarr arrays are padded past the end of the episode with zeros. If the
    # total_frames truncation in load_wrist() is wrong, every episode gets a
    # metres-per-frame jump at the boundary -> a fake impulse in ~100% of them.
    if "manifest_frames" in ok:
        m = ok.dropna(subset=["manifest_frames", "n_frames"])
        if len(m):
            match = (m["n_frames"].astype(int) == m["manifest_frames"].astype(int))
            hard(match.mean() > 0.98, "frame counts match the manifest",
                 f"{match.mean():.1%} match — mismatch means padding was NOT stripped")
            if match.mean() <= 0.98:
                d = m[~match].head(3)[["episode_id", "n_frames", "manifest_frames"]]
                print("   examples:\n   ", d.to_string(index=False).replace("\n", "\n    "))
    else:
        soft(False, "manifest_frames column present",
             "can't verify chunk-padding — re-run with the merge in main()")

    # ---- 3. degenerate output -------------------------------------------
    fs = pd.to_numeric(ok["failure_score"], errors="coerce")
    hard(fs.notna().mean() > 0.9, "failure_score populated",
         f"{fs.isna().mean():.1%} null")
    hard(fs.nunique() > 1, "scores vary", f"{fs.nunique()} distinct values")
    flag = ok["failure_flag"].astype(bool)
    prev = flag.mean()
    soft(0.001 < prev < 0.60, "prevalence is plausible", f"{prev:.1%} flagged")
    if prev >= 0.60:
        print("   -> almost everything flagged. Classic symptom of the padding "
              "bug or a z_thresh that didn't survive real data.")
    if prev <= 0.001:
        print("   -> almost nothing flagged. Check that wrist keypoints are "
              "actually varying (see the all-zero check below).")

    # ---- 4. did we read real trajectories -------------------------------
    if "nan_frac" in ok:
        nf = pd.to_numeric(ok["nan_frac"], errors="coerce")
        soft(nf.median() < 0.5, "NaN fraction sane", f"median {nf.median():.2f}")
    dead = (pd.to_numeric(ok.get("n_impulses", 0), errors="coerce").fillna(0) == 0) & \
           (pd.to_numeric(ok.get("eye_n_cycles", 0), errors="coerce").fillna(0) == 0) & \
           (fs.fillna(0) == 0)
    soft(dead.mean() < 0.5, "episodes have signal",
         f"{dead.mean():.1%} scored completely flat (possible all-zero keypoints)")

    # ---- 5. channel coverage --------------------------------------------
    if "eye_n_cycles" in ok:
        got_eye = pd.to_numeric(ok["eye_n_cycles"], errors="coerce").fillna(0) >= 3
        soft(got_eye.mean() > 0.2, "eye channel usable",
             f"{got_eye.mean():.1%} of episodes segmented >=3 cycles")
    if "rf_small_ratio" in ok:
        got_rf = pd.to_numeric(ok["rf_small_ratio"], errors="coerce").notna()
        soft(got_rf.mean() > 0.5, "rainflow channel usable",
             f"{got_rf.mean():.1%} produced a ratio")

    # ---- 6. rate-normalised prevalence (the number C actually reports) ---
    if {"manifest_frames", "fps", "lab"} <= set(ok.columns):
        g = ok.copy()
        g["minutes"] = pd.to_numeric(g["manifest_frames"], errors="coerce") / \
                       pd.to_numeric(g["fps"], errors="coerce") / 60.0
        g["flag"] = flag
        by = g.groupby("lab").agg(episodes=("episode_id", "size"),
                                  ep_prevalence=("flag", "mean"),
                                  median_min=("minutes", "median"),
                                  fails_per_min=("flag", "sum"))
        by["fails_per_min"] = by["fails_per_min"] / g.groupby("lab")["minutes"].sum()
        print("\n== prevalence by data source ==")
        print(by.to_string(float_format=lambda v: f"{v:.4f}"))
        if by["median_min"].max() > 3 * by["median_min"].min() and len(by) > 1:
            print("\n  NOTE: episode lengths differ >3x across sources. Report "
                  "fails_per_min, not ep_prevalence — the per-episode rate is "
                  "measuring duration, not quality.")

    return finish()


def finish():
    print()
    if HARD:
        print(f"{len(HARD)} HARD failure(s) — do not build on this parquet:")
        for h in HARD:
            print("  -", h)
        return 1
    print("all hard checks passed" + (f" ({len(SOFT)} warning(s))" if SOFT else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
