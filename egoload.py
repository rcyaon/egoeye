"""
egoload — one place that knows how to open an EgoVerse episode zarr.

Split out of audit_modal.py because the multimodal detector needs the head pose,
the full 21-joint hands and the camera intrinsics, not just the wrist — and the
local scripts, the Modal workers and the demo figures should not each grow their
own copy of the R2 handshake and the padding fix.

Three things the raw zarr bites you on (all still true):
  1. It is Cloudflare R2, not S3 — s3fs needs the R2 endpoint and region "auto",
     and R2 rejects an STS session token (400). Do not send R2_SESSION_TOKEN.
  2. `obs_keypoints` is stored FLAT as (T, 63); reshape row-major to (T, 21, 3).
  3. Arrays are chunk-padded past the end of the episode with ZEROS. attrs
     `total_frames` is authoritative — the pad rows are a metres-per-frame jump
     that fakes an impulse in every single episode.
"""
from __future__ import annotations

import os

HEAD_KEY = "obs_head_pose"
KP_KEY = "{hand}.obs_keypoints"
HANDS = ("left", "right")


def r2_filesystem():
    import s3fs
    endpoint = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
    return s3fs.S3FileSystem(
        key=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": endpoint, "region_name": "auto"},
    )


def open_root(zarr_path: str):
    """Open an episode zarr from R2 (s3://...) or from a local directory."""
    import zarr
    if not str(zarr_path).startswith("s3://"):
        return zarr.open(zarr_path, mode="r")
    store = zarr.storage.FsspecStore(
        r2_filesystem(), path=str(zarr_path).replace("s3://", "").rstrip("/"))
    return zarr.open(store, mode="r")


def load_episode(zarr_path: str, want_images: bool = False) -> dict:
    """Load every channel the multimodal detector uses, trimmed to total_frames.

    Returns a dict with:
      hands      {"left"/"right": (T, 21, 3)}  world-frame keypoints
      head       (T, 7) xyz + wxyz quaternion, world frame
      fps, n_frames, task, task_description, episode_id
      K          (3, 4) camera intrinsics, or None
      image_wh   (W, H)
      annotations  [{"text", "start_idx", "end_idx"}, ...]
      images     raw JPEG bytes per frame, only when want_images
    """
    import json
    import numpy as np

    root = open_root(zarr_path)
    attrs = dict(root.attrs)
    n = int(attrs.get("total_frames", 0)) or None

    def get(key):
        if key not in root:
            return None
        a = np.asarray(root[key])
        return a[:n] if n else a

    out = {
        "episode_id": attrs.get("episode_id", ""),
        "fps": float(attrs.get("fps", 30.0)),
        "task": attrs.get("task_name", ""),
        "task_description": attrs.get("task_description", ""),
        "objects": attrs.get("objects", []),
        # Present on every episode and always True on human data — a curation
        # flag ("this recording shows the task"), NOT a fumble label. Carried
        # through so nobody has to re-check that it is useless as ground truth.
        "task_success": attrs.get("task_success"),
        "task_confidence": attrs.get("task_confidence"),
    }

    head = get(HEAD_KEY)
    if head is None:
        raise KeyError(f"missing {HEAD_KEY}")
    out["head"] = head
    out["n_frames"] = len(head)

    hands = {}
    for h in HANDS:
        a = get(KP_KEY.format(hand=h))
        if a is not None:
            hands[h] = a.reshape(len(a), 21, 3) if a.ndim == 2 else a
    if not hands:
        raise KeyError("no hand keypoints in episode")
    out["hands"] = hands

    intr = attrs.get("intrinsics") or {}
    cam = next(iter(intr), None)
    out["K"] = np.asarray(intr[cam], float) if cam else None
    shape = ((attrs.get("features") or {}).get(f"images.{cam}") or {}).get("shape")
    out["image_wh"] = (int(shape[1]), int(shape[0])) if shape else (640, 360)
    out["camera"] = cam

    anns = []
    if "annotations" in root:
        for raw in np.asarray(root["annotations"]):
            try:
                anns.append(json.loads(bytes(raw).decode("utf-8")))
            except Exception:
                pass
    out["annotations"] = anns

    if want_images and cam:
        out["images"] = np.asarray(root[f"images.{cam}"])[:n] if n else np.asarray(root[f"images.{cam}"])
    return out
