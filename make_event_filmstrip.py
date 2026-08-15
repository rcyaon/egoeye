"""
make_event_filmstrip.py — the validation artifact for the multimodal detector.

The team's rule is that an unvalidated detector fails "is the method
defensible", and validation here means a human watching the top calls and
counting real failures (precision@10). Watching whole clips is the slow part:
a 15 s video to adjudicate one 0.2 s event.

This renders each flagged event as a filmstrip instead — the frames either side
of the event, with the tracked hand drawn on them, above the three channel
traces that produced the call. The frames come from the episode's own JPEGs and
the keypoints are projected with the episode's own intrinsics, so what you see
is exactly what the detector saw.

    python make_event_filmstrip.py --task wash_dishes --top 10
    python make_event_filmstrip.py --episodes-list ids.txt --outdir strips/

Needs R2 credentials in the environment (set -a; . ~/.egoverse_env; set +a).
"""
from __future__ import annotations

import argparse
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bodykit as bk
from egoload import load_episode
from eyekit import clean_trajectory, kinematics, detect_impulses

# MediaPipe hand skeleton (see bodykit convention 4)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def decode(buf) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(bytes(buf))).convert("RGB"))


def event_channels(kp, head, fps):
    """Recompute the three traces behind a call, for plotting."""
    ww = clean_trajectory(kp[:, bk.WRIST, :])
    _, acc_w = kinematics(ww, fps)
    _, acc_b = kinematics(bk.to_body_frame(ww, head), fps)
    from eyekit import impulse_z
    ang, pitch = bk.head_angular_velocity(head, fps)
    ap = bk.hand_aperture(kp)
    return {
        "impulse_z": impulse_z(acc_w, fps),
        "imp_world": detect_impulses(acc_w, fps, z_thresh=bk.DEFAULTS["z_impulse"])[0],
        "imp_body": detect_impulses(acc_b, fps, z_thresh=bk.DEFAULTS["z_impulse"])[0],
        "head_z": np.maximum(bk._robust_z(ang, fps), bk._robust_z(pitch, fps)),
        "aperture": ap,
        "release_z": bk._robust_z(bk.aperture_opening_rate(ap, fps), fps),
    }


def render_event(ep, hand, frame, out_path, n_frames=7, span_s=1.2, title=""):
    """One filmstrip: frames across +/-span_s/2 of the event, plus the traces."""
    kp = ep["hands"][hand]
    head, fps = ep["head"], ep["fps"]
    n = min(len(kp), len(head), len(ep["images"]))
    kp, head = kp[:n], head[:n]
    ch = event_channels(kp, head, fps)

    step = max(1, int(span_s * fps / (n_frames - 1)))
    idx = [int(np.clip(frame + (i - n_frames // 2) * step, 0, n - 1)) for i in range(n_frames)]

    u, v, z, _ = bk.project_to_camera(kp, head, ep["K"], ep["image_wh"])

    fig = plt.figure(figsize=(3.7 * n_frames, 9.4))
    gs = fig.add_gridspec(4, n_frames, height_ratios=[3.0, 1.0, 1.0, 1.0], hspace=0.45)

    for i, f in enumerate(idx):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(decode(ep["images"][f]))
        for a, b in BONES:
            if np.isfinite(u[f, a]) and np.isfinite(u[f, b]):
                ax.plot([u[f, a], u[f, b]], [v[f, a], v[f, b]], "-",
                        color="#00ff88", lw=1.4, alpha=0.85)
        ax.plot(u[f, bk.WRIST], v[f, bk.WRIST], "o", color="red", ms=5)
        ax.set_xlim(0, ep["image_wh"][0]); ax.set_ylim(ep["image_wh"][1], 0)
        ax.set_xticks([]); ax.set_yticks([])
        dt = (f - frame) / fps
        ax.set_title(f"{dt:+.2f}s" + ("  ← EVENT" if f == idx[n_frames // 2] else ""),
                     fontsize=11, color="red" if f == idx[n_frames // 2] else "black",
                     fontweight="bold" if f == idx[n_frames // 2] else "normal")

    lo, hi = max(0, frame - int(3 * fps)), min(n, frame + int(3 * fps))
    t = np.arange(lo, hi) / fps
    panels = [
        ("wrist impulse (z)", ch["impulse_z"][lo:hi], bk.DEFAULTS["z_impulse"], "#d62728"),
        ("head turn / pitch (z)", ch["head_z"][lo:hi], bk.DEFAULTS["z_head"], "#1f77b4"),
        ("hand opening rate (z)", ch["release_z"][lo:hi], bk.DEFAULTS["z_release"], "#2ca02c"),
    ]
    for r, (name, sig, thr, col) in enumerate(panels):
        ax = fig.add_subplot(gs[r + 1, :])
        ax.plot(t, sig, color=col, lw=1.2)
        ax.axhline(thr, ls="--", color="grey", lw=0.9)
        ax.axvline(frame / fps, color="red", lw=1.6, alpha=0.7)
        ax.axvspan((frame - bk.DEFAULTS["window_s"] * fps) / fps,
                   (frame + bk.DEFAULTS["window_s"] * fps) / fps,
                   color="red", alpha=0.07)
        ax.set_ylabel(name, fontsize=9)
        ax.set_xlim(t[0], t[-1])
        if r < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time in episode (s)", fontsize=9)

    seg = next((a.get("text", "") for a in ep["annotations"]
                if a.get("start_idx", -1) <= frame < a.get("end_idx", -1)), "")
    fig.suptitle(
        f"{title}\n{ep['episode_id']}  ·  {hand} hand  ·  frame {frame} "
        f"({frame/fps:.1f}s)  ·  annotated segment: \"{seg or 'n/a'}\"\n"
        f"task: {ep['task_description'][:110]}",
        fontsize=12, y=0.99)
    fig.savefig(out_path, dpi=95, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="episodes.csv")
    ap.add_argument("--task", default="wash_dishes")
    ap.add_argument("--top", type=int, default=10, help="how many events to render")
    ap.add_argument("--scan", type=int, default=200, help="episodes to scan for events")
    ap.add_argument("--outdir", default="event_strips")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.episodes, dtype={"episode_id": str})
    df = df[(df["task"].astype(str).str.lower() == args.task.lower()) &
            (df["embodiment"].astype(str).str.startswith("human"))]
    df = df[df["n_frames"] >= 200]
    df = df.sample(n=min(args.scan, len(df)), random_state=args.seed)
    print(f"scanning {len(df)} {args.task} episodes for confirmed events")

    hits = []
    for i, row in enumerate(df.to_dict("records")):
        try:
            ep = load_episode(row["zarr_path"])
        except Exception:
            continue
        for hand, kp in ep["hands"].items():
            r = bk.score_episode_multimodal(row["episode_id"], kp, ep["head"],
                                            ep["fps"], hand=hand, K=ep["K"],
                                            image_wh=ep["image_wh"])
            for f in r.confirmed_frames:
                hits.append({"episode_id": row["episode_id"], "zarr_path": row["zarr_path"],
                             "hand": hand, "frame": f, "score": r.failure_score,
                             "n_confirmed": r.n_confirmed})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(df)} episodes, {len(hits)} events so far")

    if not hits:
        print("no confirmed events found — nothing to render")
        return 0
    h = pd.DataFrame(hits).sort_values(["score", "n_confirmed"], ascending=False)
    h.to_csv(os.path.join(args.outdir, "events.csv"), index=False)
    print(f"\n{len(h)} confirmed events; rendering top {min(args.top, len(h))}")

    for rank, (_, e) in enumerate(h.head(args.top).iterrows(), start=1):
        ep = load_episode(e["zarr_path"], want_images=True)
        out = os.path.join(args.outdir, f"{rank:02d}_{e['episode_id']}_{e['hand']}.png")
        render_event(ep, e["hand"], int(e["frame"]), out,
                     title=f"#{rank} — multimodal failure call (score {e['score']:.2f})")
        print(f"  wrote {out}")

    print(f"\nAdjudicate each strip Y/N and record precision@{args.top} — "
          f"index in {args.outdir}/events.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
