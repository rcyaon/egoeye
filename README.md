# egoeye — signal-integrity analysis of human demonstrations

Deterministic failure detection + curation for EgoVerse episodes. No training,
no GPU, no LLM judge. Wrist kinematics → impulse detection (drops), rainflow
cycle decomposition (retries), eye-diagram mask testing (behavioral consistency).

Those three names are borrowed from signal integrity, fatigue engineering and
machine diagnostics. **If they mean nothing to you, read the next section —
it explains them from scratch, no background assumed.**

**Thesis:** most curation filters bad *recordings* (blur, dedup); this filters
bad *demonstrations* — visually perfect episodes where the human fumbled, which
is exactly the data that poisons imitation learning.

## What's an "eye diagram"? (no background needed)

The name and the method are borrowed from high-speed electronics. If you've not
seen one before, here's the whole idea.

**The problem it was invented for.** A cable carrying billions of bits per
second sends one bit every fixed slice of time — say every 100 picoseconds. The
receiver has to decide "was that a 1 or a 0?" in the middle of each slice. You
want to know: is this link reliable? You can't answer that by staring at a
waveform with a billion bits in it.

**The trick.** Chop the signal into those fixed time slices — one per bit — and
draw every slice *on top of every other slice*, all overlapping on the same
axes. Thousands of bits, stacked into one picture. Because the shape repeats,
the overlaid traces pile up into a pattern with a hole in the middle that looks
like an eye:

    consistent bits, overlaid          inconsistent bits, overlaid
    -> a clear opening: "the eye"      -> the opening closes up

       \\\\            ////               \\  \  //  \ / /
        \\\\          ////                 \ \\/ \\/ / \/
         \\\\        ////                   \/ \\ / \/\ \
          \\\\ OPEN ////                     /\  \/  / \/
         ////        \\\\                   /\ // \/\ /\
        ////          \\\\                 / //\ /\ \\ \
       ////            \\\\               / /  \/  \  \\

    time within one bit period ->      time within one bit period ->

**Reading it.** A wide-open eye means every bit looked essentially the same:
consistent timing, consistent amplitude. A smeared, half-shut eye means the bits
vary — some arrive early, some late, some weak — and the receiver will
eventually misread one. Engineers quantify this as the **eye opening** (how much
clear space is left in the middle) and by **mask testing**: draw a forbidden
shape in the centre of the eye, and if any trace intrudes into it, the link
fails. Crucially, you never have to look at any individual bit. The overlay
turns "is this thing consistent?" into a shape you can measure.

**What we do with it.** A person washing dishes is also doing a repeating thing:
scrub, rinse, set down, reach for the next one. So we treat one motion cycle the
way an engineer treats one bit period. `eyekit.py` cuts the wrist-speed trace
into individual cycles at the quiet moments between them, stretches them all to
the same length, and overlays them.

A demonstrator who does the task the same way every time produces a tight,
open eye. Someone who hesitates, fumbles, re-grips or has to redo a motion
produces cycles that don't line up — a smeared eye — and the cycle that
contains the fumble is the one that pokes through the mask. That gives a
per-episode consistency score without labelling anything, training anything,
or watching the video.

The other two borrowed terms in the same sentence, briefly: **impulse detection**
is the same maths used to hear a cracked bearing in a spinning machine — a sharp
broadband knock standing out against smooth background motion, which is what a
drop or a collision looks like. **Rainflow counting** comes from metal-fatigue
engineering, where it tallies how many large and small stress cycles a part went
through; here it separates big intentional motions from the small back-and-forth
wobbles of a correction or a re-grasp.

**One honest caveat**, covered in detail further down: on real data the rainflow
channel turned out to be saturated and the impulse channel alone flags forceful
*intentional* motion rather than failure. The multimodal detector below is the
response to that.

## Setup

Episode zarrs live in the Cloudflare R2 bucket `rldb`. Run the EgoVerse repo's
`egomimic/utils/aws/setup_secret.sh` to get `~/.egoverse_env`, then:

    set -a; . ~/.egoverse_env; set +a
    modal secret create egoverse-aws \
      R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID \
      R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \
      R2_ENDPOINT_URL=$R2_ENDPOINT_URL

Do **not** pass `R2_SESSION_TOKEN`. R2 rejects `X-Amz-Security-Token` with a 400;
the env file ships one because it is for S3, not R2.

`episodes.csv` (one row per episode: `episode_id`, `zarr_path`, `fps`,
`n_frames`, `lab`, `task`, `embodiment`) is exported from the SQL episode table —
see the EgoVerse repo's `sql_tutorial.ipynb`.

Note on filtering: `embodiment` is now `human_bimanual` and similar; the data
source moved to a separate `lab` column, so `embodiment=='aria'/'mecka'` matches
zero rows and the built-in `aria-*` / `mecka-*` download filters are dead.

## Usage

    python test_synthetic.py            # wrist-only detector, ground-truth smoke test
    python test_multimodal.py           # multimodal gate, ground-truth smoke test
    python test_search.py               # 36 checks on search parsing and ranking

    python explore_episode.py s3://rldb/processed_v3/<lab>/<id>.zarr   # inspect one episode

    modal run audit_modal.py --episodes episodes.csv --task wash_dishes --limit 50
    modal run audit_multimodal.py --episodes episodes.csv --task wash_dishes

    python make_event_filmstrip.py --task wash_dishes --top 10   # adjudicate calls
    python validate.py                                           # weak labels + top-10 gallery
    python make_demo_figs.py                                     # eye diagrams, confidence meter

Both smoke tests run offline with no credentials and print `PASS`.

## Quality-aware search

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
  — a 3.8× gap that was measuring length, not quality. It is the same length
  bias that makes the raw per-episode flag rate climb from 16% to 97% purely
  with episode duration, since impulses fire at a roughly constant background
  rate. The ranking divides by duration and the gap closes to 1.4×;
  `--success-scale raw` restores the old behaviour. `failure_score` and
  `failure_flag` are untouched — this is the ranking's channel, not a second
  detector.

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

## Multimodal detector

Wrist trajectory alone has a ceiling: impulse magnitude flags forceful
*intentional* motion — soap squirts, disposal scrapes — and video validation put
it at precision@10 = 1/10. The fix is to stop asking how hard the impulse was
and start asking whether anything else agrees with it.

    failure = wrist impulse
              AND it survives the body-motion veto
              AND within ±0.5s BOTH a head turn/pitch spike AND a hand opening

`bodykit.py` adds head/gaze, hand aperture and body-frame kinematics from
channels that were already in every zarr and unused: `obs_head_pose`, the other
20 hand keypoints, and the camera intrinsics. `eyekit.py` is untouched, so the
frozen wrist-only detector still runs unchanged alongside it.

Validation without labels uses a **circular-shift null** — shift a supporting
channel inside its own episode, which keeps its event count and burstiness and
destroys only its alignment:

| split | size | confirmed | null | enrichment | flag rate |
|---|---|---|---|---|---|
| wash_dishes, tuned on | 400 eps / 133 min | 14 | 2.4 | 5.96× | 3.0% |
| wash_dishes, held out | 200 eps / 63 min | 8 | 1.5 | 5.25× | 3.5% |
| fold_clothes, never tuned on | 243 eps / 112 min | 10 | 1.6 | 6.45× | 3.3% |

Flag rate falls 23% → ~3%, stable across two tasks. Three things to know before
quoting it:

- **AND, not OR.** OR confirms 81% of impulses and enriches 1.9×; AND enriches
  5–7×. It is also the better physics — letting go opens the hand *and* pulls
  the gaze.
- **The gate's channels share no inputs, on purpose.** Body-frame wrist
  acceleration is built from the head rotation and gaze-hand angle from both, so
  pairing either with a head channel correlates head motion with itself
  (measured: 6.3× inflation vs 2.9–3.8× for honest pairs). The gate uses the
  world-frame impulse, head-only channels and the aperture.
- **~3% is not yet a failure-prevalence number.** The gate finds co-timed
  impulse + release + gaze events; a deliberate set-down looks like that too.
  Enrichment proves the co-timing is real, not what it is. `make_event_filmstrip.py`
  renders each call as frames + traces so precision@10 can be scored quickly —
  that is the open job. See `NEXT_DIRECTION.md`.

Data conventions verified while building this, three of which contradict older
comments in the repo: keypoints reshape **row-major** (the transpose gives a
1.6 m "hand"), quaternions are **wxyz** (0.00° error vs 17–35°), and the hand is
**MediaPipe-ordered with the thumb at joints 1–4**, not MANO. Full table with the
checks in `NEXT_DIRECTION.md`.

## Files
- `eyekit.py` — core library (all the math, heavily commented with the why)
- `bodykit.py` — multimodal channels: head/gaze, hand aperture, body frame, the gate
- `egoload.py` — one loader for every episode channel (R2 handshake, padding fix)
- `audit_multimodal.py` — Modal fan-out for the multimodal gate + shuffle control
- `make_event_filmstrip.py` — per-event filmstrip for scoring precision@10
- `explore_episode.py` — schema discovery + per-episode diagnostic plots
- `audit_modal.py` — parallel dataset-wide audit on Modal
- `validate.py` — weak-label agreement + precision@10 gallery
- `make_demo_figs.py` — eye diagrams + confidence meter for the slide (reads local
  zarrs or R2 `s3://` paths)
- `make_filmstrip.py` — preview-MP4 frames laid over the confidence trace, aligned
  to timestamp — the actual photographed demo moment
- `export_annotations.py` — annotations.csv from `segments[].label` (real per-clip
  text, e.g. "drop orange bowl...") rather than just task_description
- `local_audit.py` — same scorer as `audit_modal.py`, run sequentially without
  Modal; good for a few hundred episodes when you want real numbers without a
  fan-out (e.g. to feed `egosearch.py --demo` with populated signal/success)
- `test_synthetic.py` — ground-truth smoke test (`python test_synthetic.py` → PASS)
- `egosearch.py` — quality-aware natural-language search (library + CLI)
- `search_page.py` — renders the search demo page; engine-parity checked in-browser
- `test_search.py` — 36 checks on parsing, channels and ranking (`→ PASS`)
- `make_search_targets.py` / `validate_labels.py` — scoped audit manifest + the
  free-ground-truth check

## Thresholds, and why they sit where they do

Every threshold is set once and frozen, rather than tuned per run.

- impulse `z_thresh=10`: clean synthetic motion maxes at ~5, drops hit 30–50
- rainflow noise floor 0.1×max range: below this is tracking noise (~90% of raw cycles)
- rainflow small band [0.1, 0.5)×max: corrections are sub-half-amplitude motions
- eye active-phase mask 0.25×median-max: spread is only meaningful where motion happens

Multimodal gate (`bodykit.py`), chosen against the circular-shift null and then frozen:

- `z_head=5`, `z_release=5`: on a plateau, not a spike — neighbouring values
  score 6.0–7.5× where the chosen point scores 7.1×
- `window_s=0.5`: half a second beats a full second at every other setting,
  matching the lag profile, whose peak decays inside ~0.5 s
- `edge_s=1.0`: events within 1 s of a clip boundary are discarded. Smoothing is
  ill-conditioned there, `np.gradient` falls back to one-sided differences, and
  clips often open mid-motion with the camera still settling. Before the guard,
  36% of confirmed events sat inside that first or last second; after it, 0%.
