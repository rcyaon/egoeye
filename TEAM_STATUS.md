# Team status — handoff from Person C (Midhat)

Branch drop for the team to cross-check and merge. Everything below is either
**VERIFIED on live data** (keep it) or **NEEDS YOUR CALL** (your lane, your judgment).
Cross-check your section, keep what's right, tell me what's wrong.

---

## Shared verified facts (affect everyone — please don't re-derive)

1. **Embodiment collapse (07/08/2026).** `embodiment` is now `human_bimanual` etc.;
   the data source moved to a new **`lab`** column. `embodiment=='aria'/'mecka'` return
   **0 rows** — so the built-in `aria-*` / `mecka-*` download filters are DEAD. Use `lab`.
2. **R2 access: do NOT pass the session token.** Cloudflare R2 rejects `X-Amz-Security-Token`
   (400 Bad Request). `~/.egoverse_env` ships an `R2_SESSION_TOKEN` but it's for S3 — drop it.
   Fixed in `audit_modal.py`.
3. **Wrist keypoints are flat `(T, 63)`**, not `(T, 21, 3)` → reshape then joint 0.
   `WRIST_KEY = right.obs_keypoints` **confirmed on real data**. Fixed in `make_demo_figs.py`.
4. **No ground-truth labels:** `eval_success` is empty. Validation = keyword weak-labels +
   human precision@10 only (no `eval_success` axis).
5. **Corpus:** `human_bimanual` = 414,511 eps / 355M frames (82% of data). Audit ran on a
   stratified 4,422-ep sample; **0 errors**, ~2 min. Pipeline is proven end-to-end.

---

## Person A — pipeline & run
**Status: unblocked, most of hour-0 already done.**
- ✅ Modal secret `egoverse-aws` created; smoke test **17/17**; `timeout=300` already set.
- ✅ `episodes.csv` generated from SQL with `lab`,`task` columns (`episodes_hero.csv`,
  `episodes_audit_strat.csv`). Full corpus list is `episodes_audit.csv` (414k, regen from SQL).
- ✅ Cost datapoint: 4,422 eps ≈ 2 min.
- **YOUR CALL:** launch the full/bounded fan-out — **but gate on B's recalibration** (below).

## Person B — detector  ⬅ **the gate, and the biggest open item**
**Status: NEEDS YOUR CALL. We did the diagnosis; the fix is yours.**
- 🔴 **`z_thresh=10` does NOT survive real data.** `failure_flag = (n_impulses>=1)` is
  **length-biased**: impulses fire at a ~constant background rate (~1.3/1000 frames), so
  flag rate climbs **16% → 97%** purely with episode length. The raw "62% prevalence" is a
  length artifact, not a failure rate.
- **Fix direction (yours to decide):** raise `z`, and/or require impulse *density* not raw
  count, and/or magnitude above per-episode background. Re-freeze once, then validate top-10.
- **Interaction:** your "score both wrists, take max" bimanual fix makes the length bias
  *worse* (two chances per episode) — design the two changes together.

## Person C — prevalence & slide  (this handoff)
**Status: tooling built, swap-ready for B's number.**
- ✅ `prevalence.py` — reports per-episode flag rate (length-biased) AND **failure
  events/minute** (length-normalized headline). Chart: `prevalence_by_source.png`.
- ✅ **Scoping call, verified:** `wash_dishes` is **100% microagi** (no cross-source axis).
  Best second axis = **`fold_clothes` (12,212 eps, 6 labs)** or `cup_on_saucer` (5,961, 5 labs) —
  cleaner than the messy 72k "dish family."
- Swap in B's recalibrated parquet → chart + headline auto-update (number is not hardcoded).

## Person D — demo & ground truth
**Status: figs half-built, handed over.**
- ✅ `make_demo_figs.py` flat-63 bug fixed; `confidence_meter.png` + `eye_diagrams.png`
  generated from **real** hero episodes.
- **YOUR CALL:** `annotations.csv` export (episode_id, text, preview_mp4) for `validate.py`;
  filmstrip from preview MP4 + confidence trace; add the R2 loader to `make_demo_figs.py`
  (it currently reads local zarrs only).

---

## Cut order if we fall behind (from the plan, kept)
semantic search → cross-source breakdown → filmstrip → rainflow/eye channels.
**Never cut precision@10.**
