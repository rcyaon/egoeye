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

# Same defaults as audit_modal.py's load_wrist() — kept in sync by hand since
# this script runs locally (not on Modal) and can't share that module's
# in-container-only import.
WRIST_KEY = "right.obs_keypoints"
LEFT_KEY = WRIST_KEY.replace("right.", "left.")


_R2_FS = None  # one S3FileSystem per process — reuse so we don't juggle
                # multiple async event loops (aiobotocore errors at exit otherwise)


def _r2_filesystem():
    global _R2_FS
    if _R2_FS is None:
        import os
        import s3fs

        endpoint = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
        if not endpoint:
            raise RuntimeError(
                "R2_ENDPOINT_URL not set — run `set -a; . ~/.egoverse_env; set +a` "
                "(from ./egomimic/utils/aws/setup_secret.sh in the EgoVerse repo) first."
            )
        _R2_FS = s3fs.S3FileSystem(
            key=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            client_kwargs={"endpoint_url": endpoint, "region_name": "auto"},
        )
    return _R2_FS


def _open_zarr_root(path):
    """zarr.open() for a local path or an s3://<bucket>/... R2 URI.

    R2 access mirrors audit_modal.py's load_wrist(): needs R2_ACCESS_KEY_ID /
    R2_SECRET_ACCESS_KEY / R2_ENDPOINT_URL in the environment (source
    ~/.egoverse_env first — see EgoVerse README's setup_secret.sh). Cloudflare
    R2 rejects a session token (400 Bad Request), so it is deliberately never
    sent, same as the Modal side.
    """
    if not str(path).startswith("s3://"):
        import zarr
        return zarr.open(path, mode="r")

    import zarr
    fs = _r2_filesystem()
    return zarr.open(
        zarr.storage.FsspecStore(fs, path=path.replace("s3://", "").rstrip("/")),
        mode="r",
    )


def load(path, key, joint):
    root = _open_zarr_root(path)
    key = key or (WRIST_KEY if WRIST_KEY in root else LEFT_KEY)
    arr = np.asarray(root[key])
    # obs_keypoints ship FLAT (T, 63) — reshape to (T, 21, 3) before joint select.
    if arr.ndim == 2 and arr.shape[1] > 3 and arr.shape[1] % 3 == 0:
        arr = arr.reshape(len(arr), -1, 3)
    arr = arr[:, joint, :] if arr.ndim == 3 else arr
    # Arrays are chunk-padded past episode end with zeros; zarr.json's
    # total_frames is authoritative (repo changelog) — trim it like
    # audit_modal.py does, or a demo figure gets a fake end-of-episode impulse.
    n = int(root.attrs.get("total_frames", len(arr)))
    return arr[:n]

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
    ap.add_argument("--key", default=None,
                    help="zarr wrist key; default auto-picks right/left.obs_keypoints")
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
    if _R2_FS is not None:
        # s3fs/aiobotocore's atexit session-close crosses event loops and
        # prints a scary (but harmless — exit code is still 0) traceback on
        # interpreter teardown. Both PNGs are already flushed to disk above,
        # so just skip normal teardown instead of chasing an upstream bug.
        import sys
        sys.stdout.flush()
        import os
        os._exit(0)
