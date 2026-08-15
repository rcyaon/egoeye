# Next direction — couple wrist motion with body coordinates (multimodal)

**Status: PARKED here after determining wrist-only can't do it. Everything current is
pushed. Resume from this doc.**

## What we determined (so nobody re-runs it)
1. Impulse detector flags forceful **intentional** motion, not failure — video-validated:
   precision@10 = 1/10 (original), 1/5 (after B's recalibration). Same false positives:
   soap squirt, garbage disposal, filling a bowl.
2. **Tested the impulse+correction fix** on 15 human-labeled clips → precision **2/13**,
   no lift. The "correction" wiggle after an impulse is kinematically identical to normal
   busy hand motion in cluttered tasks (nearly every impulse showed ≥3 speed reversals
   within 1.5s, real or not). **→ wrist trajectory alone has a real ceiling.**

## Why multimodal is the way forward
A genuine drop/slip leaves a signature **beyond the wrist** that intentional forceful
motion does not:
- **Gaze / head shift** — the person looks at the dropped object (`obs_head_pose`).
- **Hand opening** — the grasp releases (full MANO keypoints, not just joint 0).
A soap squirt or disposal dump has the wrist impulse **without** the gaze snap + hand-open.

## Channels already in every episode's zarr (unused so far)
- `obs_head_pose` (7) — head/gaze proxy
- `left/right.obs_keypoints` (63 = 21×3 MANO) — full hand pose / **aperture** (we only used joint 0)
- `left/right.obs_ee_pose`, `left/right.obs_wrist_pose` (7)

## Proposed gate for whoever picks this up
`failure = wrist impulse  AND  (head-pose angular-jerk spike  OR  hand-aperture jump)  within ~1s`
- head angular velocity: quaternion deltas of `obs_head_pose`
- hand aperture: spread/extent of the 21 MANO keypoints; a release = sudden increase
- validate the same way we did: watch the new top-10, count real failures

## Artifacts to resume from
- Labeled clips + verdicts: `~/Documents/Hackathon/clips/`, `clips_recal/` (+ scorecards)
- wash_dishes scores: `audit_washdishes.parquet`, `audit_wd_recal.parquet`
- Correction-gate test (the negative result): `scratchpad/test_correction.py`
- Findings: `VALIDATION_FINDINGS.md`  ·  Slide: `SLIDE.md`
