# Multimodal detector — built, measured, and where it now stands

**Status: the multimodal direction from the last handoff is IMPLEMENTED and
validated against a shuffle control. It is not yet validated against human eyes
— that is the one remaining job, and the tool to do it fast is written.**

Previous state of this doc (kept, still true): wrist-only hit a ceiling.
Impulse magnitude flags forceful *intentional* motion, precision@10 = 1/10, and
the impulse+correction fix gave no lift on 15 labelled clips.

---

## 1. What the data actually looks like (verified, do not re-derive)

The last handoff listed the unused channels but not their conventions. All five
of these were checked against real episodes; three contradict what the older
comments in the repo assume.

| thing | answer | how it was checked |
|---|---|---|
| `obs_keypoints` (T,63) layout | **row-major (T,21,3)** — the transpose is wrong | row-major gives a 0.148 m hand, the (3,21) transpose gives a 1.62 m "hand" |
| quaternion order in every `*_pose` | **wxyz** (scalar first) | rebuilt the hand frame from keypoints: wxyz reproduces the stored rotation to **0.00°**, xyzw drifts 17–35° |
| coordinate frame | keypoints, wrist/ee pose and head pose share **one world frame** | keypoint joint 0 equals `obs_wrist_pose[:,:3]` to 0.00000 m |
| hand joint order | **MediaPipe, thumb = joints 1–4** — *not* MANO | 18/20 bones have cv = 0.0000 under that chain; thumb base sits 0.040 m from the wrist vs 0.083–0.097 m for finger MCPs |
| head camera vs head pose | **no axis remap** — optical x-right / y-down / z-forward | projected the keypoints into the JPEG with `attrs["intrinsics"]`; they land on the hand |

That last one is the "take the pixel data into account" note, and it works:
the hand is inside the head camera's view **99.8% of frames**, so gaze geometry
is well posed. `attrs["intrinsics"]` is a 3×4 for `front_1`, 640×360, ~105° HFOV.

Two more facts worth having:

- **`attrs` carries `task_success`, but it is not ground truth.** It is `True`
  on 80/80 sampled human episodes — a curation flag meaning "this recording
  shows the task", not "the human did it cleanly". Same conclusion the team
  reached about `eval_success`, reached a different way. Do not build an
  evaluation on it.
- **`annotations` is richer than the SQL text**: a JSON array of
  `{text, start_idx, end_idx}` segments, e.g. `{"text":"apply soap to sponge",
  "start_idx":0,"end_idx":183}`. Per-segment, frame-indexed. `egoload.py`
  parses it; the filmstrip prints the segment a flagged event falls in.

## 2. The detector

`bodykit.py`. eyekit.py is untouched, so the frozen wrist-only detector still
runs exactly as before and the two can be compared on the same episode.

    failure = wrist impulse (z>10, world frame)
              AND it survives the body-motion veto
              AND it is NOT within 1 s of either end of the clip
              AND within ±0.5 s BOTH:
                    a head turn/pitch spike (z>5)   [head pose only]
                    a hand-opening spike   (z>5)    [hand keypoints only]

Three design points that are load-bearing:

- **AND, not OR.** The last handoff proposed `impulse AND (head OR hand-open)`.
  OR is barely selective — it confirms 81% of impulses and enriches only 1.9×
  over the null. AND enriches 5–7×. It is also the better physics: letting go
  of a plate opens the hand *and* pulls the gaze.
- **The channels in the gate share no inputs.** This is a correctness
  constraint. Body-frame wrist acceleration is built *from* the head rotation,
  and gaze-hand angle is built from head *and* hand — pairing either with a
  head channel correlates head motion with itself. Measured: the honest pairs
  peak at 2.9–3.8× their own baseline, the circular pair inflates to 6.3×.
  So the gate uses the **world-frame** impulse (hand only) with head channels
  (head only) and aperture (hand, non-wrist joints). `gaze_hand_angle` is
  reported but deliberately kept out of the gate.
- **The body frame is a veto, not a gate partner.** "Was that the hand or the
  whole person moving" costs no coincidence budget.

## 3. What it measures

Validation without labels uses a **circular-shift null**: shift a supporting
channel by a random offset inside its own episode, which preserves its event
count and burst structure exactly and destroys only its alignment. Then
`enrichment = confirmations(real) / confirmations(shifted)`. A uniform-random
null cannot do this — it under-counts coincidences for bursty channels and
reports lift where there is none.

| split | size | impulses | confirmed | null | enrichment | flag rate |
|---|---|---|---|---|---|---|
| wash_dishes, all (tuned on) | 400 eps / 133 min | 99 | 14 | 2.4 | **5.96×** (z=6.0) | 3.0% |
| wash_dishes, held-out half | 200 eps / 63 min | 44 | 8 | 1.5 | **5.25×** (z=4.8) | 3.5% |
| **fold_clothes, never tuned on** | 243 eps / 112 min | 110 | 10 | 1.6 | **6.45×** (z=6.6) | 3.3% |

(Minutes are of video. Both hands are scored separately, so that is twice as
many episode-hand traces — 265 min for the first row.)

Independently, the lag profile — coincidence rate as a function of time offset —
peaks sharply at τ=0 and decays inside ~0.5 s, on channel pairs that share no
inputs. That is genuine event-locking, not two busy channels overlapping.

Flag rate goes **23% (wrist-only) → ~3%**, and the ~3% is stable across two
tasks and both halves of the tuning task.

## 4. What this does NOT yet establish — read before quoting a number

**The gate finds co-timed impulse + release + gaze events. Nobody has yet
confirmed those are failures.** The specific worry, visible in the first
filmstrips: a *deliberate set-down* also involves contact, a hand opening and
a gaze shift. One top call sits in a segment annotated "place bowl in cabinet",
which may well be a clean placement. Enrichment over a shuffle control proves
the co-timing is real; it says nothing about what the co-timed thing *is*.

So: **do not report ~3% as a failure-prevalence number yet.** It is the rate of
confirmed release-with-impact events.

## 5. The next job, and the tool for it

`make_event_filmstrip.py` renders each flagged event as frames either side of it
with the tracked hand drawn on, above the three channel traces that produced the
call, plus the annotated segment text. It exists because adjudicating a 0.2 s
event by watching a 15 s clip is the slow step in precision@10.

    set -a; . ~/.egoverse_env; set +a
    python make_event_filmstrip.py --task wash_dishes --top 10 --scan 250

Score each strip Y/N → precision@10 for the multimodal gate, directly comparable
to the 1/10 the wrist-only detector got. That single number decides whether this
is a detector or just a well-controlled coincidence statistic.

If the failures turn out to be dominated by deliberate placements, the next
discriminator to try is what distinguishes a place from a drop: a placement
decelerates before contact and the hand withdraws calmly, a drop does not
decelerate and is followed by a fast re-grab toward the release point. Test it
the same way — enrichment against the shuffle null first, then filmstrips.

## 6. Running it at scale

    modal run audit_multimodal.py --episodes episodes.csv --task wash_dishes --limit 50
    modal run audit_multimodal.py --episodes episodes.csv --task wash_dishes

Smoke-tested at 40 episodes, 0 errors. It runs the circular-shift control as a
second fan-out by default and prints real vs shifted side by side —
**report the two together or the prevalence number means nothing.**
`--min-frames 200` skips clips too short to survive the 1 s edge guard.

## 7. Files

- `bodykit.py` — the multimodal channels, the gate, the null. All conventions
  documented at the top with how each was verified.
- `egoload.py` — one loader for every channel (head, both hands, intrinsics,
  annotations, images), R2 handshake and the zero-padding fix in one place.
- `audit_multimodal.py` — Modal fan-out + the shuffle control.
- `make_event_filmstrip.py` — the precision@10 artifact.
- `eyekit.py`, `audit_modal.py` — untouched, still the frozen wrist-only path.
