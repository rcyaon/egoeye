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

## Files
- `eyekit.py` — core library (all the math, heavily commented with the why)
- `explore_episode.py` — hour-0 schema discovery + go/no-go plots
- `audit_modal.py` — parallel dataset-wide audit on Modal
- `validate.py` — weak-label agreement + precision@10 gallery
- `make_demo_figs.py` — eye diagrams + confidence meter for the slide
- `test_synthetic.py` — ground-truth smoke test (`python test_synthetic.py` → PASS)

## Known thresholds to defend under questioning
- impulse `z_thresh=10`: clean synthetic maxes at ~5, drops hit 30–50 (huge margin; re-verify on 3 real episodes)
- rainflow noise floor 0.1×max range: below this is tracking noise (~90% of raw cycles)
- rainflow small band [0.1, 0.5)×max: corrections are sub-half-amplitude motions
- eye active-phase mask 0.25×median-max: spread is only meaningful where motion happens
