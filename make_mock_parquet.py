"""Generate a fake audit_results.parquet with the REAL schema.

    python make_mock_parquet.py --n 4000 --out mock_results.parquet

Why: the Modal fan-out is the only serial step in the project, and the
prevalence chart, the slide and the validation script all just read its output.
Build them against this, then swap in the real parquet — same columns, same
dtypes, no code changes.

The numbers are invented. The SHAPE is deliberately realistic: prevalence
varies by lab, and episode durations differ ~8x across labs, so anything that
naively reports per-episode rates will look wrong here in exactly the way it
would look wrong on real data.
"""

import argparse
import numpy as np
import pandas as pd

DISH_FAMILY = (r"dish|wash_pot|wash_pan|wash_glass|wash_frying_pan|"
               r"wash_kitchen_utensil|wash_the_pot")

# invented per-lab failure rates, so the by-source breakdown has real contrast
LAB_PREVALENCE = {"microagi": 0.11, "mecka": 0.26, "scale": 0.18}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="episodes.csv")
    ap.add_argument("--out", default="mock_results.parquet")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--error-rate", type=float, default=0.02)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    df = pd.read_csv(args.episodes, dtype={"episode_id": str})
    tasks = df["task"].astype(str).str.lower()
    df = df[tasks.str.contains(DISH_FAMILY, regex=True, na=False)]
    df = df[df["embodiment"].astype(str).str.startswith("human")]
    df = df.sample(n=min(args.n, len(df)), random_state=0).reset_index(drop=True)

    n = len(df)
    p = df["lab"].map(LAB_PREVALENCE).fillna(0.15).to_numpy()
    is_fail = rng.random(n) < p

    # failure episodes: >=1 impulse and a high score; clean: 0 impulses, low score
    n_imp = np.where(is_fail, rng.poisson(1.4, n) + 1, 0)
    score = np.where(is_fail,
                     np.clip(rng.normal(0.72, 0.14, n), 0.30, 1.0),
                     np.clip(np.abs(rng.normal(0.04, 0.06, n)), 0.0, 0.45))

    frames = df["n_frames"].to_numpy(dtype=float)
    fps = df["fps"].to_numpy(dtype=float)
    dur_s = frames / fps

    # the eye channel needs >=3 segmented cycles; short episodes just don't get it
    eye_cycles = np.clip((dur_s / 2.5).astype(int) + rng.integers(-1, 2, n), 0, 40)
    has_eye = eye_cycles >= 3
    eye_open = np.where(has_eye,
                        np.clip(rng.normal(0.62, 0.13, n) - 0.10 * is_fail, 0, 1),
                        np.nan)

    rf_small = np.clip(rng.normal(0.10, 0.07, n) + 0.28 * is_fail, 0, 1)
    rf_small[rng.random(n) < 0.06] = np.nan          # too few cycles to decompose

    res = pd.DataFrame({
        "episode_id": df["episode_id"],
        "n_frames": frames.astype(int),
        "nan_frac": np.clip(np.abs(rng.normal(0.02, 0.03, n)), 0, 1),
        "n_impulses": n_imp,
        "impulse_frames": [sorted(rng.integers(0, max(int(f), 2), k).tolist())
                           for f, k in zip(frames, n_imp)],
        "rf_small_ratio": rf_small,
        "rf_n_cycles": np.clip(rng.normal(dur_s * 1.6, 6), 0, None).round(),
        "eye_opening": eye_open,
        "eye_n_cycles": np.where(has_eye, eye_cycles, 0).astype(int),
        "mask_violation_p90": np.clip(rng.normal(0.12, 0.05, n), 0, 1),
        "failure_flag": is_fail | (score >= 0.5),
        "failure_score": score,
        "error": "",
    })

    # a realistic slice of failures, so error handling gets exercised downstream
    bad = rng.random(n) < args.error_rate
    res.loc[bad, "error"] = "FileNotFoundError('zarr key right.obs_keypoints')"
    # In a real run these keys are simply absent on error rows, so pandas builds
    # the column with NaNs and widens the dtype. Reproduce that, or downstream
    # code passes here and breaks on the real parquet.
    for c in ["failure_flag", "failure_score", "n_impulses", "eye_opening",
              "rf_small_ratio", "nan_frac", "eye_n_cycles", "n_frames"]:
        res[c] = res[c].astype(object if c == "failure_flag" else float)
        res.loc[bad, c] = np.nan

    # the merge audit_modal.main() does after the fan-out
    res = res.merge(
        df[["episode_id", "lab", "task", "embodiment", "fps", "n_frames"]]
          .rename(columns={"n_frames": "manifest_frames"}),
        on="episode_id", how="left")

    res.to_parquet(args.out)
    print(f"wrote {args.out}: {len(res)} rows, {int(bad.sum())} error rows")
    print(f"labs: {res['lab'].value_counts().to_dict()}")
    print(f"mock prevalence: {res[res.error=='']['failure_flag'].astype(bool).mean():.1%}")
    print("\nNOTE: these numbers are INVENTED. Never put them on a slide.")


if __name__ == "__main__":
    main()
