"""Score the detector against outcomes the dataset already wrote down.

    python validate_labels.py search_audit.parquet

Complements validate.py rather than replacing it. validate.py checks agreement
with annotation *text* and asks a human to watch the top 10. This checks
against task *names* that state the outcome — bag_groceries_success,
cup_on_saucer_failure — which nobody on this team authored and nobody tuned a
threshold against.

Read the caveat in the output before quoting the number. On the human episodes
the labels are one-sided (all _success), so what this measures is the
false-positive rate: how often the detector calls a failure on an episode the
dataset itself calls clean. That is the number most exposed to z_thresh=10
having been frozen on synthetic data.
"""

import sys

import pandas as pd

LABEL = r"_success|_failure|success$|failure$"


def main(path="search_audit.parquet"):
    res = pd.read_parquet(path)
    if "error" in res:
        res = res[res["error"].fillna("") == ""]
    if "task" not in res:
        print("no task column — re-run the audit with the manifest merge in "
              "audit_modal.main()")
        return 1

    task = res["task"].fillna("").astype(str).str.lower()
    lab = res[task.str.contains(LABEL, regex=True, na=False)].copy()
    if not len(lab):
        print(f"no outcome-labelled episodes in {path}.\n"
              "Build the manifest first: python make_search_targets.py")
        return 1

    lab["truth_fail"] = lab["task"].str.lower().str.contains("failure")
    lab["ours_fail"] = lab["failure_flag"].astype(bool)
    n_ok = int((~lab["truth_fail"]).sum())
    n_bad = int(lab["truth_fail"].sum())

    print(f"== {len(lab)} outcome-labelled episodes "
          f"({n_ok} labelled success, {n_bad} labelled failure) ==\n")
    print(pd.crosstab(lab["ours_fail"], lab["truth_fail"],
                      rownames=["we flagged"], colnames=["name says failure"]))

    if n_ok:
        fp = float(lab.loc[~lab["truth_fail"], "ours_fail"].mean())
        print(f"\nfalse-positive rate on known-good episodes: {fp:.1%} "
              f"({int(fp * n_ok)}/{n_ok})")
        if fp > 0.30:
            print("  -> the detector runs hot on real footage. z_thresh was "
                  "frozen on synthetic data; this is the evidence to revisit it.")
        else:
            print("  -> holds up: the threshold survived contact with episodes "
                  "nobody tuned it against.")
        clean = lab.loc[~lab["truth_fail"], "failure_score"]
        print(f"  failure_score on known-good: median {clean.median():.2f}, "
              f"p90 {clean.quantile(0.9):.2f}")

    if n_bad:
        rec = float(lab.loc[lab["truth_fail"], "ours_fail"].mean())
        print(f"\nrecall on known-failure episodes: {rec:.1%} "
              f"({int(rec * n_bad)}/{n_bad})")
    else:
        print("\nCAVEAT: every labelled episode here is a success. This is a "
              "false-positive rate, NOT precision or recall — do not put "
              "'accuracy' on a slide. The catalogue's _failure episodes are "
              "all robot embodiments (see make_search_targets.py).")

    worst = lab[~lab["truth_fail"]].nlargest(5, "failure_score")
    if len(worst):
        print("\nlabelled-success episodes we scored worst — watch these first:")
        cols = [c for c in ["episode_id", "task", "failure_score", "n_impulses",
                            "eye_opening"] if c in worst]
        print(worst[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
