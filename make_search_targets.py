"""Build a scoped audit manifest that makes the search demo work.

    python make_search_targets.py            # -> search_targets.csv
    modal run audit_modal.py --episodes search_targets.csv --out search_audit.parquet
    python egosearch.py --demo --results search_audit.parquet --html demo_search.html

The problem this solves: the fan-out is scoped to the dish family, so every
quality-aware query outside it returns episodes marked UNAUDITED. The search
demo works, but its whole point — that the audit re-orders the results — is
invisible for the brief's own flagship example ("placing a cup into a drawer").

Two tiers, and the second one is the interesting one:

  1. The flagship family: every task mentioning a cup, a drawer or a saucer,
     capped per task so one 6.6k-episode task cannot eat the run.

  2. FREE GROUND TRUTH — 93 episodes with the outcome written into the task
     name (bag_groceries_success, cup_on_saucer_success, fold_clothes_success).
     Nobody labelled these for us and nothing in the repo uses them.

     Read the split before believing anything about it: all 93 human-embodiment
     labelled episodes are _success. Every _failure episode in the catalogue
     (170 of them) is eva_bimanual — robot teleop, not a human demonstration.
     So this is NOT precision and recall. It is a one-sided specificity check,
     and that happens to be the exact failure mode this detector is most
     exposed to: z_thresh=10 was frozen on synthetic data, and if it runs hot
     on real footage it will flag episodes the dataset itself calls clean.
     93 free negatives measure that directly.

     --include-robot-labelled adds the 1,492 eva episodes (1,322 success / 170
     failure), which would give a genuine two-class set. Gated behind a flag
     because nobody has confirmed eva_bimanual zarrs expose the same
     right.obs_keypoints key — check one episode before spending the run.
"""

import argparse
import re

import pandas as pd

# the brief's flagship query is "placing a cup into a drawer"
FLAGSHIP = r"drawer|saucer|\bcup\b|cup_|_cup"
# outcome written into the task slug
LABELLED = r"_success|_failure|success$|failure$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="episodes.csv")
    ap.add_argument("--out", default="search_targets.csv")
    ap.add_argument("--per-task", type=int, default=60,
                    help="cap per distinct task for the flagship family")
    ap.add_argument("--max", type=int, default=2500)
    ap.add_argument("--include-robot-labelled", action="store_true",
                    help="add the 1,492 eva_bimanual labelled episodes — only "
                         "useful if their zarrs expose the same wrist key")
    args = ap.parse_args()

    df = pd.read_csv(args.episodes, dtype={"episode_id": str})
    df["task"] = df["task"].fillna("").astype(str)
    task = df["task"].str.lower()

    # eyekit reads MANO wrist keypoints; the eva_*/yam_* embodiments are robots
    human = df["embodiment"].astype(str).str.startswith("human")

    is_labelled = task.str.contains(LABELLED, regex=True, na=False)
    labelled = df[(human | args.include_robot_labelled) & is_labelled]
    flag_pool = df[human & task.str.contains(FLAGSHIP, regex=True, na=False)]
    flagship = (flag_pool.sort_values("episode_id")
                         .groupby("task", group_keys=False)
                         .head(args.per_task))

    out = pd.concat([labelled, flagship]).drop_duplicates("episode_id")
    if len(out) > args.max:
        # keep every labelled episode; trim only the flagship family
        keep_lab = out[out["episode_id"].isin(labelled["episode_id"])]
        rest = out[~out["episode_id"].isin(labelled["episode_id"])]
        out = pd.concat([keep_lab, rest.head(max(0, args.max - len(keep_lab)))])

    out = out.sort_values("episode_id")
    out.to_csv(args.out, index=False)

    minutes = (out["n_frames"] / out["fps"]).sum() / 60
    print(f"wrote {args.out}: {len(out)} episodes, {minutes:.0f} min of video")
    print(f"  labelled ground truth : {len(labelled)}")
    print(f"  flagship family       : {len(out) - len(labelled)}")
    print(f"  labs                  : {out['lab'].value_counts().to_dict()}")
    print(f"  tasks                 : {out['task'].nunique()}")
    lab_tasks = labelled["task"].str.lower()
    n_ok = int(lab_tasks.str.contains("success").sum())
    n_bad = int(lab_tasks.str.contains("failure").sum())
    print(f"\nground-truth split: {n_ok} success / {n_bad} failure")
    if n_bad == 0:
        print("  one-sided: this measures the false-positive rate only "
              "(see --include-robot-labelled)")
    print("after the run: python validate_labels.py search_audit.parquet")


if __name__ == "__main__":
    main()
