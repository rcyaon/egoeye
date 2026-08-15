# Submission slide — skeleton (Person C)

**Live slide (screenshot into the deck):**
https://claude.ai/code/artifact/8e0a6bc8-5670-43e5-96e6-07d0c7806c33

Reframed around the video-validation result (see `VALIDATION_FINDINGS.md`), not a raw
prevalence number — because the raw number doesn't survive a judge watching one clip.

## What's on it (honest skeleton)
1. **Method** — wrist keypoints (both hands) → kinematics → impulse z-score →
   **failure = impulse + correction** (the recovery signature, not impulse alone).
2. **Validation** — precision@10 = **1/10** on wash_dishes, watched on video.
   1 real (dish fell → correction) vs 9 forceful-intentional motions (soap squirt,
   disposal scrape). This is the honest, defensible core.
3. **Confidence meter** — real episode, trace dips at the event.
4. **Proof strip** — audited on Modal, 0 errors, ~2 min, pennies; bimanual (left hand
   worse ~45%); validated on video, not just synthetic; deterministic, CPU-only, no LLM.

## Pending slots (fill as info lands)
- precision@10 number: re-validating the recalibrated detector now (top-5 re-watch).
  If it climbs, swap it in. If not, 1/10 stands and the honest framing holds.
- Optional: if B lands the impulse+correction gate and precision improves, promote it
  from "next step" to "result."

## Rebuild
`build_slide.py` (in scratchpad) embeds `confidence_meter.png` as a data URI and writes
the HTML. Regenerate the PNG with `make_demo_figs.py` first if it's missing.
