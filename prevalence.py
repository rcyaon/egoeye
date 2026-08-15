"""
prevalence.py — Person C's headline number + cross-source chart.

Reads the audit parquet (from audit_modal.py) + episode metadata, and reports
failure prevalence TWO ways:

  1. per-episode flag rate      — the intuitive number, BUT length-biased
  2. failure EVENTS per minute  — length-normalized, the DEFENSIBLE headline

Why per-minute is the headline (measured, not assumed): flag rate rises 16%%->97%%
purely with episode length because impulses occur at a ~constant background rate.
Mecka clips (median ~84s) vs microagi (~11s) would make mecka look ~8x worse for
mechanical reasons alone. Events-per-minute removes that.

    python prevalence.py --results audit_strat.parquet --meta episodes_audit_strat.csv
    # swap in the real/recalibrated parquet when B re-freezes — same schema, no code change.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Coarse task families (NEVER fine task family — many are n<=2, §8 #5).
FAMILY_RULES = [
    ("dishes",  "wash|dish|plate|cup|saucer|cutlery|utensil"),
    ("laundry", "fold|laundry|iron|clothes|towel"),
    ("food",    "dough|vegetable|food|meal|salad|cook|bake|pastry|potato|pepper|ingredient"),
    ("pack",    "pack|sort|container|box|bag"),
]
def task_family(t: str) -> str:
    t = str(t).lower()
    for name, pat in FAMILY_RULES:
        if pd.Series([t]).str.contains(pat, regex=True).iloc[0]:
            return name
    return "other"


def load(results: str, meta: str) -> pd.DataFrame:
    df = pd.read_parquet(results)
    df = df[df.get("error", "") == ""].copy()
    m = pd.read_csv(meta)[["episode_id", "lab", "task"]]
    df["episode_id"] = df["episode_id"].astype(str)
    m["episode_id"] = m["episode_id"].astype(str)
    df = df.merge(m, on="episode_id", how="left")
    df["minutes"] = df["n_frames"] / 30.0 / 60.0
    df["family"] = df["task"].map(task_family)
    return df


def _rates(g: pd.DataFrame) -> pd.Series:
    mins = g["minutes"].sum()
    return pd.Series({
        "episodes": len(g),
        "flag_rate_pct": 100 * g["failure_flag"].mean(),          # length-biased
        "events_per_min": g["n_impulses"].sum() / max(mins, 1e-9), # defensible
        "median_dur_s": 60 * g["minutes"].median(),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="audit_strat.parquet")
    ap.add_argument("--meta", default="episodes_audit_strat.csv")
    ap.add_argument("--axis", default="lab", choices=["lab", "family"],
                    help="cross-source axis for the chart")
    args = ap.parse_args()

    df = load(args.results, args.meta)
    overall = _rates(df)
    print(f"=== AUDIT: {len(df)} episodes scored ===")
    print(f"per-episode flag rate : {overall['flag_rate_pct']:.1f}%  (LENGTH-BIASED — do not headline)")
    print(f"failure events/minute : {overall['events_per_min']:.2f}  (headline metric)")
    print()

    by = df.groupby(args.axis).apply(_rates, include_groups=False).sort_values(
        "events_per_min", ascending=False)
    print(f"=== by {args.axis} ===")
    print(by.round(2).to_string())

    # Chart: events/minute by source (length-robust). Flag rate shown faded for contrast.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(by))
    ax.bar(x, by["events_per_min"], color="#c0392b", label="failure events / min (defensible)")
    ax.set_xticks(x); ax.set_xticklabels(by.index, rotation=30, ha="right")
    ax.set_ylabel("failure events per minute")
    ax.set_title(f"Failure-event rate by {args.axis} — length-normalized")
    ax2 = ax.twinx()
    ax2.plot(x, by["flag_rate_pct"], "o--", color="#7f8c8d", alpha=0.6,
             label="per-episode flag rate (length-biased)")
    ax2.set_ylabel("per-episode flag rate (%)", color="#7f8c8d")
    ax.legend(loc="upper right"); ax2.legend(loc="upper center")
    fig.tight_layout(); fig.savefig("prevalence_by_source.png", dpi=140)
    print("\nwrote prevalence_by_source.png")

    top = by.index[0]
    overall_epm = overall["events_per_min"]
    top_epm = by.loc[top, "events_per_min"]
    print(f'\nHEADLINE SENTENCE (auto): "Failure events occur at {overall_epm:.1f}/min '
          f'across the corpus, concentrated in {top} ({top_epm:.1f}/min)."')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
