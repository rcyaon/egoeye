"""
Two figures per chosen episode:
  1. the eye diagram (tight vs smeared, side by side if you pass two episodes)
  2. confidence-meter trace to play under the preview MP4

Usage:
  python make_demo_figs.py clean_ep.zarr smeared_ep.zarr --key <wrist key> --fps 30
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eyekit import (clean_trajectory, kinematics, segment_cycles,
                    eye_matrix, eye_metrics, confidence_trace)

def load(path, key, joint):
    import zarr
    arr = np.asarray(zarr.open(path, mode="r")[key])
    # obs_keypoints ship FLAT (T, 63) — reshape to (T, 21, 3) before joint select.
    if arr.ndim == 2 and arr.shape[1] > 3 and arr.shape[1] % 3 == 0:
        arr = arr.reshape(len(arr), -1, 3)
    return arr[:, joint, :] if arr.ndim == 3 else arr

def plot_eye(ax, xyz, fps, title):
    speed, _ = kinematics(clean_trajectory(xyz), fps)
    M = eye_matrix(speed, segment_cycles(speed, fps))
    if M is None:
        ax.set_title(f"{title} (no cycles)"); return
    for row in M:
        ax.plot(row, alpha=0.12, color="C0")
    ax.plot(np.median(M, axis=0), color="C1", lw=2.5)
    m = eye_metrics(M)
    ax.set_title(f"{title}\neye opening = {m['eye_opening']:.2f}  "
                 f"({m['eye_n_cycles']} cycles)")
    ax.set_xlabel("phase"); ax.set_ylabel("normalized wrist speed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+")
    ap.add_argument("--key", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--joint", type=int, default=0)
    args = ap.parse_args()

    n = len(args.episodes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, ep in zip(axes[0], args.episodes):
        plot_eye(ax, load(ep, args.key, args.joint), args.fps, ep.split("/")[-1])
    fig.suptitle("Eye-diagram mask testing of human demonstrations", y=1.02)
    fig.tight_layout(); fig.savefig("eye_diagrams.png", dpi=150, bbox_inches="tight")
    print("wrote eye_diagrams.png")

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    t, conf = confidence_trace(load(args.episodes[-1], args.key, args.joint), args.fps)
    ax2.plot(t, conf, lw=2)
    ax2.fill_between(t, conf, 1.0, alpha=0.2, color="C3")
    ax2.set_ylim(0, 1.05); ax2.set_xlabel("time (s)")
    ax2.set_ylabel("success confidence")
    ax2.set_title("confidence meter — dips at impulsive events (drops/collisions)")
    fig2.tight_layout(); fig2.savefig("confidence_meter.png", dpi=150)
    print("wrote confidence_meter.png — screenshot next to the preview MP4 frame")

if __name__ == "__main__":
    main()
