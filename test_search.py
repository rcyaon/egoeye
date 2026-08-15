"""Ground-truth tests for egosearch. Run: python test_search.py -> PASS.

Two layers:
  A) synthetic — a hand-built 8-episode corpus where the right answer is known
     by construction, so every ranking rule is checked in isolation.
  B) real corpus — runs only if episodes.csv is present. Checks the flagship
     query from the brief actually retrieves drawer episodes out of 438k.

Same contract as test_synthetic.py: this must print PASS before the demo goes
anywhere near a projector.
"""

import os
import sys

import numpy as np
import pandas as pd

from egosearch import (EgoSearch, parse_query, stem, signal_score,
                       success_score, tokenize)

FAILS = []


def check(ok, label, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(label)


# ----------------------------------------------------------------------
# A. tokenisation
# ----------------------------------------------------------------------
print("== tokenisation ==")
check(stem("folding") == "fold" and stem("folds") == "fold"
      and stem("dishes") == "dish", "gerunds and plurals collapse",
      f'folding->{stem("folding")} dishes->{stem("dishes")}')
check(stem("water") == "water" and stem("paper") == "paper"
      and stem("drawer") == "drawer" and stem("container") == "container",
      "-er nouns survive the stemmer",
      "the rule that would break these is deliberately absent")
check(tokenize("freeform_put_cup_on_saucer") == ["put", "cup", "saucer"],
      "snake_case splits, drops stopwords and collection prefixes",
      str(tokenize("freeform_put_cup_on_saucer")))
check("demonstration" not in tokenize("a demonstration of the episode"),
      "corpus-scaffolding words are dropped")

# ----------------------------------------------------------------------
# B. query understanding
# ----------------------------------------------------------------------
print("\n== query understanding ==")
q = parse_query("find successful demonstrations of placing a cup into a drawer")
check(q.intent == "success", "'successful' -> success intent", q.intent)
check(q.raw_terms == ["placing", "cup", "drawer"],
      "content terms survive, scaffolding does not", str(q.raw_terms))

q = parse_query("wash dishes episodes that were fumbled")
check(q.intent == "failure", "'fumbled' -> failure intent", q.intent)

q = parse_query("packing groceries without dropping anything")
check(q.intent == "success", "negated failure word flips to success",
      f"{q.intent} via {q.intent_evidence}")

q = parse_query("fold clothes with no mistakes")
check(q.intent == "success", "'no mistakes' flips to success", q.intent)

q = parse_query("folding clothes from mecka")
check(q.lab == "mecka" and "mecka" not in q.raw_terms,
      "lab filter parsed behind a preposition and removed from terms",
      f"lab={q.lab} terms={q.raw_terms}")

q = parse_query("scale the recipe up")
check(q.lab is None, "bare 'scale' stays a content word, not a lab filter",
      f"lab={q.lab}")

q = parse_query("lab:scale washing dishes")
check(q.lab == "scale", "explicit lab: operator", f"lab={q.lab}")

q = parse_query("wash dishes under 20 seconds")
check(q.max_dur == 20.0 and q.min_dur is None, "max duration parsed",
      f"max={q.max_dur}")
q = parse_query("wash dishes longer than 2 minutes")
check(q.min_dur == 120.0, "min duration parsed with unit conversion",
      f"min={q.min_dur}")
q = parse_query("short bimanual kitchen demos")
check(q.max_dur is not None and q.embodiment == "bimanual",
      "'short' and embodiment parsed together",
      f"max={q.max_dur} emb={q.embodiment}")
q = parse_query("top 3 successful wash dishes")
check(q.k == 3, "result count parsed", f"k={q.k}")

# ----------------------------------------------------------------------
# C. quality channels
# ----------------------------------------------------------------------
print("\n== quality channels ==")
clean_row = {"eye_opening": 0.85, "mask_violation_p90": 0.05,
             "rf_small_ratio": 0.04, "nan_frac": 0.01, "failure_score": 0.02}
dirty_row = {"eye_opening": 0.20, "mask_violation_p90": 0.28,
             "rf_small_ratio": 0.40, "nan_frac": 0.20, "failure_score": 0.90}
check(signal_score(clean_row) > 0.8 > signal_score(dirty_row),
      "signal separates clean from dirty",
      f"{signal_score(clean_row):.2f} vs {signal_score(dirty_row):.2f}")
check(abs(success_score(clean_row) - 0.98) < 1e-9,
      "success = 1 - failure_score", f"{success_score(clean_row):.2f}")

partial = {"eye_opening": np.nan, "mask_violation_p90": np.nan,
           "rf_small_ratio": 0.04, "nan_frac": 0.01}
check(np.isfinite(signal_score(partial)) and signal_score(partial) > 0.8,
      "signal degrades gracefully when the eye channel is absent",
      f"{signal_score(partial):.2f}")
check(not np.isfinite(signal_score({"failure_score": 0.1})),
      "no channels at all -> NaN, never 0",
      "'unmeasured' must not read as 'terrible'")

# ----------------------------------------------------------------------
# D. ranking on a synthetic corpus with known answers
# ----------------------------------------------------------------------
print("\n== ranking ==")


def corpus():
    """8 episodes: 3 wash_dishes at graded quality, 1 unaudited wash_dishes,
    2 fold_clothes, 1 pristine-but-irrelevant, 1 mecka wash_dishes."""
    rows = [
        # id          task            lab        eye   mask  rf    nan  fail  scored
        ("clean",  "wash_dishes",  "microagi", 0.90, 0.04, 0.03, 0.01, 0.02, 1),
        ("mid",    "wash_dishes",  "microagi", 0.55, 0.15, 0.18, 0.05, 0.30, 1),
        ("fumble", "wash_dishes",  "microagi", 0.20, 0.29, 0.45, 0.10, 0.95, 1),
        ("unaud",  "wash_dishes",  "microagi", 0.00, 0.00, 0.00, 0.00, 0.00, 0),
        ("fold_a", "fold_clothes", "mecka",    0.80, 0.05, 0.05, 0.01, 0.05, 1),
        ("fold_b", "fold_clothes", "mecka",    0.30, 0.25, 0.35, 0.08, 0.80, 1),
        ("shiny",  "assemble_blender", "scale", 0.99, 0.01, 0.01, 0.00, 0.00, 1),
        ("mecka_d", "washing_dishes", "mecka", 0.70, 0.10, 0.10, 0.02, 0.10, 1),
    ]
    df = pd.DataFrame(rows, columns=[
        "episode_id", "task", "lab", "eye_opening", "mask_violation_p90",
        "rf_small_ratio", "nan_frac", "failure_score", "_scored"])
    df["_scored"] = df["_scored"].astype(bool)
    df.loc[~df["_scored"], ["eye_opening", "mask_violation_p90",
                            "rf_small_ratio", "nan_frac", "failure_score"]] = np.nan
    df["embodiment"] = "human_bimanual"
    df["fps"] = 30.0
    df["n_frames"] = 300.0
    df["duration_s"] = 10.0
    df["text"] = ""
    df["zarr_path"] = ""
    df["impulse_frames"] = [[] for _ in range(len(df))]
    df["_doc"] = df["task"].str.replace("_", " ", regex=False)
    return EgoSearch(df)


ix = corpus()

_, hits = ix.search("successful wash dishes demonstrations", k=8)
order = [h.episode_id for h in hits]
check(order[0] == "clean", "success intent puts the cleanest episode first",
      str(order[:4]))
check(order.index("clean") < order.index("mid") < order.index("fumble"),
      "quality ordering is monotone within the same task", str(order[:4]))
check(order.index("clean") < order.index("unaud") < order.index("fumble"),
      "unaudited episodes rank between measured-clean and measured-fumbled",
      str(order[:4]))

_, hits = ix.search("wash dishes that were fumbled", k=8)
order = [h.episode_id for h in hits]
check(order[0] == "fumble", "failure intent inverts the quality ordering",
      str(order[:3]))

_, hits = ix.search("successful wash dishes", k=8)
check("shiny" not in [h.episode_id for h in hits[:3]],
      "a pristine but irrelevant episode never outranks relevant ones",
      "quality multiplies relevance, it does not substitute for it")

_, hits = ix.search("successful wash dishes", k=8, quality_weight=0.0)
sems = [h.semantic for h in hits]
check(sems == sorted(sems, reverse=True),
      "quality_weight=0 degenerates to pure semantic search",
      "the ablation a judge will ask for")

_, hits = ix.search("clean examples of folding clothes from mecka", k=8)
check([h.episode_id for h in hits] == ["fold_a", "fold_b"],
      "lab filter restricts the pool and quality orders what is left",
      str([h.episode_id for h in hits]))

_, hits = ix.search("successful dish washing", k=3)
check(all(h.task.replace("_", " ").find("dish") >= 0 for h in hits),
      "'dish washing' retrieves wash_dishes via stemming + word order",
      str([h.task for h in hits]))

_, hits = ix.search("successful demonstrations of cleaning plates", k=3)
check(len(hits) > 0 and "dish" in hits[0].task,
      "synonym expansion: cleaning plates -> wash_dishes",
      str([h.task for h in hits[:2]]))

_, hits = ix.search("scrubbing dishes", k=8)
tasks = {h.task for h in hits}
check(tasks and tasks <= {"wash_dishes", "washing_dishes"},
      "synonym expansion does not leak into unrelated tasks", str(sorted(tasks)))

q = parse_query("cleaning the kitchen")
check(q.intent == "none" and "cleaning" in q.raw_terms,
      "'cleaning the kitchen' is a task, not a quality filter",
      f"intent={q.intent} terms={q.raw_terms}")
q = parse_query("clean demonstrations of cleaning the kitchen")
check(q.intent == "success" and "cleaning" in q.raw_terms,
      "the same word can be intent once and a term once",
      f"intent={q.intent} terms={q.raw_terms}")

# ----------------------------------------------------------------------
# E. real corpus (skipped if episodes.csv is absent)
# ----------------------------------------------------------------------
print("\n== real corpus ==")
if not os.path.exists("episodes.csv"):
    print("SKIP  episodes.csv not present")
else:
    results = ("mock_results.parquet" if os.path.exists("mock_results.parquet")
               else None)
    real = EgoSearch.build("episodes.csv", results, None, scope="all",
                           verbose=False)
    q, hits = real.search(
        "find successful demonstrations of placing a cup into a drawer", k=5)
    tasks = [h.task.lower() for h in hits]
    check(any("drawer" in t for t in tasks),
          "flagship query retrieves drawer episodes out of 438k",
          str(tasks[:2]))

    q, hits = real.search("washing dishes from mecka without drops", k=5)
    check(all(h.lab == "mecka" for h in hits) and
          all("dish" in h.task.lower() for h in hits),
          "lab filter + negated failure word on the real corpus",
          f"{hits[0].task} / {hits[0].lab}" if hits else "no hits")

    if results:
        q, hits = real.search("wash dishes that were fumbled", k=10)
        scored = [h for h in hits if h.scored]
        check(len(scored) >= 8 and np.mean([h.success for h in scored]) < 0.3,
              "failure query surfaces measured-bad episodes",
              f"mean success={np.mean([h.success for h in scored]):.2f}")

        q, hits = real.search("clean examples of washing dishes", k=10)
        scored = [h for h in hits if h.scored]
        check(len(scored) >= 8 and np.mean([h.success for h in scored]) > 0.9,
              "success query surfaces measured-good episodes",
              f"mean success={np.mean([h.success for h in scored]):.2f}")

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("PASS")
