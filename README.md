# egoeye — signal-integrity analysis of human demonstrations

Deterministic failure detection + curation for EgoVerse episodes. No training,
no GPU, no LLM judge. Wrist kinematics → impulse detection (drops), rainflow
cycle decomposition (retries), eye-diagram mask testing (behavioral consistency).

**Thesis:** most curation filters bad *recordings* (blur, dedup); this filters
bad *demonstrations* — visually perfect episodes where the human fumbled, which
is exactly the data that poisons imitation learning.

## Status

- Core pipeline tested end-to-end on synthetic ground truth: **PASS**
  (mean failure score 0.00 clean vs 0.78 fumbled, 6/6 flagged, 0/6 false alarms)
- `sample_output.png` shows what the demo figures look like
- **One TODO before real data:** the zarr wrist-keypoint key name
  (fill `WRIST_KEY` in `audit_modal.py` after running `explore_episode.py`)

## Run order (= hour plan)

| Hour | Do | File |
|---|---|---|
| 0–1 | Pull 3 episodes (EgoVerse README: AWS configure + `sync_s3.py --filters aria-fold-clothes`). Discover schema, confirm cyclical speed + fumble spikes. **GO/NO-GO here.** | `explore_episode.py` |
| 1–2 | Sanity-check thresholds on your 3 local episodes; tune `z_thresh` / bands ONCE, then freeze | `eyekit.py` |
| 2–3 | Export episode list from SQL table (`sql_tutorial.ipynb`) → `episodes.csv`; fill `WRIST_KEY`; smoke test `modal run audit_modal.py --limit 20` | `audit_modal.py` |
| 3–4 | Full fan-out on Modal → `audit_results.parquet` + headline prevalence number | `audit_modal.py` |
| 4–4.5 | Keyword agreement + **watch the top-10** → precision@10 | `validate.py` |
| 4.5–4:45 | Eye diagrams (tight vs smeared) + confidence meter figs → summary slide | `make_demo_figs.py` |

## Descope levers (in order)
1. Fewer episodes in the fan-out (whatever finishes, ships)
2. Skip rainflow/eye, ship impulse-only detector (it's the reliable channel)
3. Static filmstrip instead of side-by-side video demo
Never skip: the precision@10 spot-check. An unvalidated detector fails
"is the method defensible."

## Pivot (if fold-clothes fumbles don't spike in hour 1)
Same pipeline, softer claim: tag *hesitation/correction* episodes via
rainflow small-cycle bursts + eye smear instead of drops. All deliverables
survive unchanged.

## Track framing (decide at 4pm, not now)
- **Track 3:** tagged episodes + prevalence audit + confidence meter (all three deliverables from one parquet)
- **Track 1:** failure tags = drop list; prevalence audit = validation report; "optimal subset = maximally free of failure demonstrations"

## Quality-aware search (deliverable 6)

    python egosearch.py "find successful demonstrations of placing a cup into a drawer"
    python egosearch.py --demo --results audit_results.parquet --html demo_search.html

A text index answers *what an episode is of*. The audit answers *whether the
human did it cleanly*. Neither is a training-set filter alone; joined, they are.
Every hit carries three scores:

| score | source | meaning |
|---|---|---|
| semantic | BM25 over task text + annotations, domain synonyms | what it's of |
| signal | eye opening, rainflow, mask violations, tracking dropout | how clean the trace is |
| success | impulses **per minute** vs a corpus-derived reference | no drop/collision event |

`final = semantic × quality^w`. Multiplicative on purpose: quality re-orders
relevance, it never substitutes for it, so a pristine but irrelevant episode
can't surface. `w=0` is the ablation (plain semantic search) and it's a slider
on the demo page. Query parsing is rule-based — intent (`successful` /
`fumbled` / `without dropping`), lab, embodiment, duration — because an LLM
parser would reintroduce the nondeterminism this whole project argues against.

Runs against `episodes.csv` alone (quality columns just read `—`); the audit
parquet lights up the quality half. **No embedding model, no LLM, no network.**

Three things worth knowing before quoting numbers from it:

- **The success channel is length-normalised, and it has to be.** Ranking on
  raw `failure_score` sorted by episode duration: `corr(duration, n_impulses)
  = 0.66` on the audited episodes, and "clean examples of washing dishes"
  returned a mean of 9.0s against 34.6s for "wash dishes that were fumbled"
  — a 3.8× gap that was measuring length, not quality (same artifact B found
  as 16%→97% flag rate, and C normalises with failures/minute). The ranking
  divides by duration; the gap closes to 1.4×. `--success-scale raw` restores
  the old behaviour, and it should become the default again once B's
  recalibrated threshold makes the impulse count mean something on its own.
  `failure_score` and `failure_flag` are untouched — this is the ranking's
  channel, not a second detector.

- **`rf_small_ratio` is saturated on real data** — 0.61–0.95 across the first
  50 real episodes, against eyekit's `[0.05, 0.35]` mapping. It discriminates
  nothing and puts a constant 0.25 floor under every real `failure_score`.
  `egosearch.py` warns when it detects this and percentile-ranks the signal
  channel to compensate; the actual fix belongs in `eyekit.py`.
- **93 episodes carry free ground truth** in their task names
  (`bag_groceries_success`, `cup_on_saucer_success`). All human-embodiment ones
  are `_success` — every `_failure` in the catalogue is robot teleop — so they
  measure the false-positive rate, not accuracy. `make_search_targets.py`
  scopes an audit run over them plus the flagship cup/drawer family;
  `validate_labels.py` scores it.

## Files
- `eyekit.py` — core library (all the math, heavily commented with the why)
- `explore_episode.py` — hour-0 schema discovery + go/no-go plots
- `audit_modal.py` — parallel dataset-wide audit on Modal
- `validate.py` — weak-label agreement + precision@10 gallery
- `make_demo_figs.py` — eye diagrams + confidence meter for the slide
- `test_synthetic.py` — ground-truth smoke test (`python test_synthetic.py` → PASS)
- `egosearch.py` — quality-aware natural-language search (library + CLI)
- `search_page.py` — renders the search demo page; engine-parity checked in-browser
- `test_search.py` — 36 checks on parsing, channels and ranking (`→ PASS`)
- `make_search_targets.py` / `validate_labels.py` — scoped audit manifest + the
  free-ground-truth check

## Known thresholds to defend under questioning
- impulse `z_thresh=10`: clean synthetic maxes at ~5, drops hit 30–50 (huge margin; re-verify on 3 real episodes)
- rainflow noise floor 0.1×max range: below this is tracking noise (~90% of raw cycles)
- rainflow small band [0.1, 0.5)×max: corrections are sub-half-amplitude motions
- eye active-phase mask 0.25×median-max: spread is only meaningful where motion happens
