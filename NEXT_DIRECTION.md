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

## Channels already in every episode's zarr (unused so far)
- `obs_head_pose` (7) — head/gaze proxy
- `left/right.obs_keypoints` (63 = 21×3 MANO) — full hand pose / **aperture** (we only used joint 0)
- `left/right.obs_ee_pose`, `left/right.obs_wrist_pose` (7)

## Proposed gate for whoever picks this up
`failure = wrist impulse  AND  (head-pose angular-jerk spike  OR  hand-aperture jump)  within ~1s`
- head angular velocity: quaternion deltas of `obs_head_pose`
- hand aperture: spread/extent of the 21 MANO keypoints; a release = sudden increase
- validate the same way we did: watch the new top-10, count real failures

## UPDATE — multimodal tested, also no lift (2026 hackathon)
Prototyped the gate above (`scratchpad/test_multimodal.py`): wrist impulse AND a nearby
head-angular-jerk **or** hand-aperture spike, on the same 15 labeled clips.
**Result: precision 2/13 — identical to wrist-only and to impulse+correction.**
Head jerk and hand-aperture spikes fire near ~every impulse because dishwashing scenes
are constantly active in *every* channel. Three approaches now tested, all ~2/13:
wrist-impulse · impulse+correction · wrist+head+hand.

**Conclusion:** the failure event (an object visibly falling) is not cleanly present in the
available motion/pose channels for cluttered tasks — it's semantic/visual. Deterministic
kinematic detection has a real ceiling here. This is the honest, defensible result.
Multimodal *done carefully* (release-specific hand-open, gaze-locked-to-object) is real
future work, not a hackathon-timeframe win.

## Artifacts to resume from
- Labeled clips + verdicts: `~/Documents/Hackathon/clips/`, `clips_recal/` (+ scorecards)
- wash_dishes scores: `audit_washdishes.parquet`, `audit_wd_recal.parquet`
- Correction-gate test (the negative result): `scratchpad/test_correction.py`
- Findings: `VALIDATION_FINDINGS.md`  ·  Slide: `SLIDE.md`
