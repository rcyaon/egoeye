"""
Goals, in order:
  1. Discover the actual zarr schema (key names, shapes, fps).
  2. Extract wrist xyz for one episode and plot speed.
  3. GO/NO-GO: does a known-ish fumble produce an acceleration spike,
     and does fold-clothes speed look cyclical?

Usage:
    python explore_episode.py /path/to/episode.zarr           # inspect schema
    python explore_episode.py s3://bucket/path/to/episode.zarr  # inspect remote R2 zarr
    python explore_episode.py /path/to/episode.zarr --key obs/wrist_pos --fps 30

Data access: follow EgoVerse README (AWS configure step + sync_s3.py with
--filters aria-fold-clothes) to pull a few episodes locally. Also open the
repo's zarr_data_viz.ipynb + sql_tutorial.ipynb — they show the canonical
key names; paste the wrist-position key below into WRIST_KEY_GUESSES.
"""

import sys, argparse
import numpy as np

# Populate from zarr_data_viz.ipynb the moment you see real keys:
WRIST_KEY_GUESSES = [
    # e.g. "obs/ee_pose", "hand/right/wrist", "actions/xyz", "mano/joints"
]

def walk_zarr(g, prefix=""):
    import zarr
    for name, item in g.items():
        path = f"{prefix}/{name}"
        if hasattr(item, "shape"):
            print(f"  {path:<45} shape={item.shape} dtype={item.dtype}")
        else:
            print(f"  {path}/")
            walk_zarr(item, path)


def open_episode_root(path: str):
    import os
    import s3fs
    import zarr

    if path.startswith("s3://"):
        endpoint = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
        fs = s3fs.S3FileSystem(
            key=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            client_kwargs={"endpoint_url": endpoint, "region_name": "auto"},
        )
        store = zarr.storage.FsspecStore(fs, path=path.replace("s3://", "").rstrip("/"))
        return zarr.open(store, mode="r")

    return zarr.open(path, mode="r")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--key", default=None, help="zarr key holding (T,3) or (T,J,3) wrist/hand positions")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--joint", type=int, default=0, help="joint index if (T,J,3); MANO wrist is joint 0")
    args = ap.parse_args()

    root = open_episode_root(args.path)
    print(f"== schema of {args.path} ==")
    walk_zarr(root)

    key = args.key
    if key is None:
        for k in WRIST_KEY_GUESSES:
            try:
                _ = root[k]; key = k; break
            except KeyError:
                continue
    if key is None:
        print("\nNo wrist key given/found. Re-run with --key <path printed above>.")
        sys.exit(0)

    arr = np.asarray(root[key])
    n = int(root.attrs.get("total_frames", len(arr)))
    arr = arr[:n]
    if arr.ndim == 2 and arr.shape[1] > 3 and arr.shape[1] % 3 == 0:
        arr = arr.reshape(len(arr), -1, 3)
    if arr.ndim == 3:                      # (T, J, 3) -> pick wrist joint
        arr = arr[:, args.joint, :]
    assert arr.ndim == 2 and arr.shape[1] == 3, f"expected (T,3), got {arr.shape}"
    print(f"\nUsing {key}: {arr.shape}, NaN frac = {np.isnan(arr).mean():.3f}")

    from eyekit import score_episode, kinematics, clean_trajectory, segment_cycles, eye_matrix
    rep = score_episode("explore", arr, fps=args.fps)
    print("\n== episode report ==")
    for k, v in rep.to_dict().items():
        print(f"  {k}: {v}")

    # plots: the go/no-go evidence
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz = clean_trajectory(arr)
    speed, accel = kinematics(xyz, args.fps)
    segs = segment_cycles(speed, args.fps)
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    t = np.arange(len(speed)) / args.fps
    ax[0].plot(t, speed); ax[0].set_title(f"wrist speed — {len(segs)} cycles found (cyclical? if not, wrong task family)")
    for a, b in segs:
        ax[0].axvspan(a / args.fps, b / args.fps, alpha=0.08)
    ax[1].plot(t, accel); ax[1].set_title("|accel| — do fumbles spike here? (go/no-go for track 3)")
    M = eye_matrix(speed, segs)
    if M is not None:
        for row in M:
            ax[2].plot(row, alpha=0.15, color="C0")
        ax[2].plot(np.median(M, axis=0), color="C1", lw=2)
        ax[2].set_title("the eye")
    fig.tight_layout(); fig.savefig("explore.png", dpi=120)
    print("\nwrote explore.png — this picture IS the hour-1 decision.")

if __name__ == "__main__":
    main()
