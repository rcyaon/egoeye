"""
The confidence-meter filmstrip: N frames pulled from the preview MP4, laid
out above the confidence trace so each frame sits directly over its own
timestamp on the curve. This is the actual demo photo — confidence_meter.png
alone is just a line chart with no video in it.

Usage:
  python make_filmstrip.py EPISODE.zarr EPISODE_video.mp4 --key <wrist key> --fps 30
  python make_filmstrip.py s3://rldb/.../ep.zarr s3://rldb/.../ep_video.mp4 --n-frames 8

Reuses make_demo_figs.py's R2 loader (local zarr or s3:// via s3fs) so it
matches audit_modal.py's load_wrist() exactly — same WRIST_KEY guess, same
flat-63 reshape, same total_frames trim.
"""

import argparse
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eyekit import confidence_trace
import make_demo_figs
from make_demo_figs import load, _r2_filesystem


def _local_mp4_path(path: str) -> str:
    """Return a local filesystem path to the mp4, downloading from R2 first if needed.

    Plain boto3 rather than the shared s3fs instance: s3fs's async .get() run
    right after zarr's FsspecStore touches the same event loop and blows up
    with "attached to a different loop" — a real crash, not the cosmetic
    exit-time one. A synchronous boto3 client for one whole-file GET sidesteps
    the async plumbing entirely.
    """
    if not path.startswith("s3://"):
        return path
    import boto3

    bucket, key = path.replace("s3://", "").split("/", 1)
    endpoint = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name="auto",
    )
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    client.download_file(bucket, key, tmp_path)
    return tmp_path


def extract_frames(mp4_path: str, timestamps_s):
    """RGB uint8 frame (or None if the seek misses) for each timestamp in seconds."""
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {mp4_path}")
    frames = []
    for t in timestamps_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_zarr")
    ap.add_argument("preview_mp4")
    ap.add_argument("--key", default=None,
                    help="zarr wrist key; default auto-picks right/left.obs_keypoints")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--out", default="filmstrip.png")
    args = ap.parse_args()

    xyz = load(args.episode_zarr, args.key, args.joint)
    t, conf = confidence_trace(xyz, args.fps)
    duration_s = t[-1]

    # evenly spaced sample points, inset slightly from the very edges so the
    # first/last thumbnails aren't cut off by a seek landing past EOF
    frame_ts = np.linspace(0.03 * duration_s, 0.97 * duration_s, args.n_frames)

    local_mp4 = _local_mp4_path(args.preview_mp4)
    try:
        frames = extract_frames(local_mp4, frame_ts)
    finally:
        if local_mp4 != args.preview_mp4:
            os.remove(local_mp4)

    fig = plt.figure(figsize=(2.2 * args.n_frames, 4.2))
    gs = fig.add_gridspec(2, args.n_frames, height_ratios=[1.4, 1], hspace=0.15,
                          top=0.88, bottom=0.12)

    for i, (frame, ts) in enumerate(zip(frames, frame_ts)):
        ax = fig.add_subplot(gs[0, i])
        if frame is not None:
            ax.imshow(frame)
        else:
            ax.text(0.5, 0.5, "no frame", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{ts:.1f}s", fontsize=9)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, conf, lw=2, color="C0")
    ax2.fill_between(t, conf, 1.0, alpha=0.2, color="C3")
    for ts in frame_ts:
        ax2.axvline(ts, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("success\nconfidence")

    fig.suptitle("Confidence-meter filmstrip — dips align with the frame at that moment", y=0.99)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
    if make_demo_figs._R2_FS is not None:
        # same s3fs/aiobotocore exit-time event-loop crash as make_demo_figs.py;
        # frames + figure are already on disk by this point.
        import sys
        sys.stdout.flush()
        os._exit(0)
