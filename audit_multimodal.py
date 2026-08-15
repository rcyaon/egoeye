"""
Dataset-wide multimodal audit on Modal.

    modal run audit_multimodal.py --episodes episodes.csv --task wash_dishes --limit 50
    modal run audit_multimodal.py --episodes episodes.csv --task wash_dishes
    modal run audit_multimodal.py --episodes episodes.csv --shuffle-null   # the control

Sibling of audit_modal.py, which runs the frozen wrist-only detector. This one
runs the gate from bodykit.py: a wrist impulse that survives the body-motion
veto AND is joined within half a second by BOTH a head/gaze event and a hand
release. Same manifest, same secret, same fan-out shape.

--shuffle-null re-runs with each supporting channel circularly shifted inside
its own episode. That is the control the headline number needs: it preserves
every channel's event count and burst structure and destroys only the alignment,
so `real / null` says whether the confirmations are co-timed or coincidental.
Report the two together or the prevalence number means nothing.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "pandas", "rainflow", "zarr", "s3fs", "pyarrow")
    .add_local_file("eyekit.py", remote_path="/root/eyekit.py")
    .add_local_file("bodykit.py", remote_path="/root/bodykit.py")
    .add_local_file("egoload.py", remote_path="/root/egoload.py")
)
app = modal.App("egoeye-multimodal", image=image)

# Same secret as audit_modal.py — R2 access key + secret + endpoint, and NOT the
# session token (R2 rejects X-Amz-Security-Token with a 400).
secret = modal.Secret.from_name("egoverse-aws")


@app.function(secrets=[secret], timeout=900, retries=1, max_containers=100)
def audit_one(row: dict) -> dict:
    import numpy as np
    import bodykit as bk
    from egoload import load_episode
    from eyekit import score_episode

    eid = str(row.get("episode_id"))
    try:
        ep = load_episode(row["zarr_path"])
        fps = float(ep["fps"] or row.get("fps", 30.0))
        out = {"episode_id": eid, "error": "", "fps": fps,
               "task_description": ep["task_description"],
               "n_annotations": len(ep["annotations"])}

        per_hand = {}
        for hand, kp in ep["hands"].items():
            r = bk.score_episode_multimodal(eid, kp, ep["head"], fps, hand=hand,
                                            K=ep["K"], image_wh=ep["image_wh"])
            per_hand[hand] = r
            # the frozen wrist-only detector on the same data, for the delta
            old = score_episode(eid, kp[:, bk.WRIST, :], fps=fps)
            out[f"{hand}_old_flag"] = bool(old.failure_flag)
            out[f"{hand}_old_impulses"] = int(old.n_impulses)

        # An episode is flagged if EITHER hand carries a confirmed event: a drop
        # is one-handed. Summed counts, not maxed, for the same reason.
        out["n_confirmed"] = sum(r.n_confirmed for r in per_hand.values())
        out["failure_flag"] = bool(out["n_confirmed"] >= 1)
        out["failure_score"] = float(max(r.failure_score for r in per_hand.values()))
        out["old_failure_flag"] = bool(any(out.get(f"{h}_old_flag") for h in per_hand))
        for k in ("n_impulses_world", "n_impulses_hand", "n_head_events",
                  "n_release_events", "n_confirmed_head", "n_confirmed_release"):
            out[k] = int(sum(getattr(r, k) for r in per_hand.values()))
        for k in ("duration_s", "n_frames"):
            out[k] = float(max(getattr(r, k) for r in per_hand.values()))
        for k in ("body_motion_frac", "frac_hand_in_view", "gaze_hand_angle_median",
                  "aperture_median", "lift_head", "lift_release"):
            vals = [getattr(r, k) for r in per_hand.values()]
            vals = [v for v in vals if v == v]
            out[k] = float(np.mean(vals)) if vals else None
        out["confirmed_frames"] = str({h: r.confirmed_frames for h, r in per_hand.items()})
        out["hands_scored"] = ",".join(sorted(per_hand))
        return out
    except Exception as e:                     # never let one episode kill the run
        return {"episode_id": eid, "error": repr(e)}


@app.function(secrets=[secret], timeout=900, retries=1, max_containers=100)
def null_one(row: dict) -> dict:
    """Circular-shift control: identical pipeline, supporting channels shifted."""
    import numpy as np
    import bodykit as bk
    from egoload import load_episode
    from eyekit import clean_trajectory, kinematics, detect_impulses

    import zlib
    N_SHIFT = 5           # average several draws: one shift per episode is noisy

    eid = str(row.get("episode_id"))
    try:
        ep = load_episode(row["zarr_path"])
        fps = float(ep["fps"] or row.get("fps", 30.0))
        th = bk.DEFAULTS
        # crc32, not hash(): Python salts string hashing per process, so hash()
        # would make this control unreproducible between runs.
        rng = np.random.default_rng(zlib.crc32(eid.encode()))
        n_conf, n_imp = 0.0, 0
        for hand, kp in ep["hands"].items():
            head = ep["head"]
            n = min(len(kp), len(head))
            kp, head = kp[:n], head[:n]
            if n < 120:
                continue
            ww = clean_trajectory(kp[:, bk.WRIST, :])
            _, aw = kinematics(ww, fps)
            _, ab = kinematics(bk.to_body_frame(ww, head), fps)
            imp = detect_impulses(aw, fps, z_thresh=th["z_impulse"])[0]
            if len(imp):
                imp = imp[bk.coincidences(imp, detect_impulses(
                    ab, fps, z_thresh=th["z_impulse"])[0], fps, th["body_veto_s"])]
            edge = int(th["edge_s"] * fps)
            if len(imp):
                imp = imp[(imp >= edge) & (imp < n - edge)]
            if len(imp) == 0:
                continue
            n_imp += len(imp)
            ang, pitch = bk.head_angular_velocity(head, fps)
            hd = np.union1d(bk.detect_events(ang, fps, th["z_head"])[0],
                            bk.detect_events(pitch, fps, th["z_head"])[0]).astype(int)
            rl = bk.detect_events(bk.aperture_opening_rate(bk.hand_aperture(kp), fps),
                                  fps, th["z_release"])[0]
            shift = lambda ev: np.sort((ev + rng.integers(0, n)) % n) if len(ev) else ev
            for _ in range(N_SHIFT):
                sh, sr = shift(hd), shift(rl)
                ih = set(bk.coincidences(imp, sh, fps, th["window_s"])) if len(sh) else set()
                ir = set(bk.coincidences(imp, sr, fps, th["window_s"])) if len(sr) else set()
                n_conf += len(ih & ir if th["require_both"] else ih | ir) / N_SHIFT
        return {"episode_id": eid, "error": "", "n_confirmed": float(n_conf),
                "n_impulses_hand": n_imp, "failure_flag": bool(n_conf >= 1)}
    except Exception as e:
        return {"episode_id": eid, "error": repr(e)}


@app.local_entrypoint()
def main(episodes: str = "episodes.csv", limit: int = 0,
         out: str = "multimodal_results.parquet",
         task: str = "", lab: str = "", human_only: bool = True,
         min_frames: int = 200, shuffle: bool = False, shuffle_null: bool = True):
    """
      --task wash_dishes    exact task, case-insensitive
      --lab microagi        restrict to one data source
      --min-frames 200      skip very short clips (the 1s edge guard eats them)
      --shuffle             random sample rather than the first N
      --no-shuffle-null     skip the circular-shift control (not recommended)
    """
    import pandas as pd, time
    df = pd.read_csv(episodes, dtype={"episode_id": str})
    n0 = len(df)
    if task:
        df = df[df["task"].astype(str).str.lower() == task.lower()]
    if lab:
        df = df[df["lab"].astype(str).str.lower() == lab.lower()]
    if human_only:
        df = df[df["embodiment"].astype(str).str.startswith("human")]
    if min_frames:
        df = df[df["n_frames"].fillna(0) >= min_frames]
    if shuffle:
        df = df.sample(frac=1.0, random_state=0)
    if limit:
        df = df.head(limit)

    rows = df.to_dict("records")
    if not rows:
        print("no episodes matched the filters — nothing to do"); return
    print(f"multimodal audit on {len(rows)} of {n0} episodes "
          f"| labs={sorted(df['lab'].astype(str).unique())[:6]} "
          f"| tasks={df['task'].nunique()}")

    t0 = time.time()
    res = pd.DataFrame(list(audit_one.map(rows)))
    dt = time.time() - t0
    res = res.merge(df[["episode_id", "lab", "task", "embodiment"]],
                    on="episode_id", how="left")
    res.to_parquet(out)

    ok = res[res["error"] == ""] if "error" in res else res
    print(f"\ndone in {dt/60:.1f} min | {len(ok)}/{len(res)} succeeded")
    if not len(ok):
        return
    mins = ok["duration_s"].sum() / 60
    conf = ok["n_confirmed"].sum()
    print(f"\ncorpus: {mins:.0f} min of video")
    print(f"  raw wrist impulses (world)   : {ok['n_impulses_world'].sum():6.0f}")
    print(f"  after body veto + edge guard : {ok['n_impulses_hand'].sum():6.0f}")
    print(f"  head/gaze events             : {ok['n_head_events'].sum():6.0f}")
    print(f"  hand-release events          : {ok['n_release_events'].sum():6.0f}")
    print(f"  CONFIRMED by the gate        : {conf:6.0f}")
    print(f"\nflag rate, frozen wrist-only : {ok['old_failure_flag'].mean():6.1%}")
    print(f"flag rate, multimodal gate   : {ok['failure_flag'].mean():6.1%}")

    if shuffle_null:
        print("\nrunning circular-shift control...")
        nul = pd.DataFrame(list(null_one.map(rows)))
        nul = nul[nul["error"] == ""] if "error" in nul else nul
        n_null = nul["n_confirmed"].sum()
        print(f"  confirmations, real    : {conf:6.0f}")
        print(f"  confirmations, shifted : {n_null:6.0f}")
        if n_null > 0:
            print(f"  ENRICHMENT             : {conf/n_null:6.2f}x")
        else:
            print("  ENRICHMENT             :    inf (no confirmations survive shifting)")
        print(f"  flag rate under null   : {nul['failure_flag'].mean():6.1%}")

    print("\ntop calls (adjudicate these with make_event_filmstrip.py):")
    cols = [c for c in ["episode_id", "task", "lab", "failure_score", "n_confirmed",
                        "n_impulses_hand", "duration_s"] if c in ok]
    print(ok.sort_values(["n_confirmed", "failure_score"], ascending=False)
            [cols].head(10).to_string())
    bad = res[res["error"] != ""] if "error" in res else res.iloc[:0]
    if len(bad):
        print(f"\n{len(bad)} errors, most common:")
        print(bad["error"].str.slice(0, 90).value_counts().head(5).to_string())
    print(f"\nwrote {out}")
