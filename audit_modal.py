"""
    modal run audit_modal.py --episodes episodes.csv --limit 50   # smoke test
    modal run audit_modal.py --episodes episodes.csv              # full audit

episodes.csv comes from the SQL episode table (see repo's sql_tutorial.ipynb):
one row per episode with columns: episode_id, zarr_path (s3/r2 URI), fps.
Refresh Scale hashes from the SQL table — old ones are stale (repo changelog).

ONE TODO before this runs: fill in load_wrist() with the real key + storage
access you discovered in explore_episode.py. Everything else is done.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "pandas", "rainflow", "zarr", "s3fs", "pyarrow")
    .add_local_file("eyekit.py", remote_path="/root/eyekit.py")
)
app = modal.App("egoeye-audit", image=image)

# Credentials: the zarrs are NOT on plain S3 — they live in the Cloudflare R2
# bucket `rldb`, and the README's AWS keys only unlock Secrets Manager. Run the
# repo's egomimic/utils/aws/setup_secret.sh first; it writes ~/.egoverse_env
# with the real R2 keys + endpoint. Then:
#   set -a; . ~/.egoverse_env; set +a
#   modal secret create egoverse-aws \
#     R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID \
#     R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \
#     R2_ENDPOINT_URL=$R2_ENDPOINT_URL
secret = modal.Secret.from_name("egoverse-aws")

WRIST_KEY = "right.obs_keypoints"   # canonical MANO keypoints, flat (T, 21*3)
WRIST_JOINT = 0                      # MANO joint 0 = wrist
LEFT_KEY = WRIST_KEY.replace("right.", "left.")  # left-only episodes (human_left_arm)


def load_wrist(zarr_path: str):
    """Return (T,3) wrist positions for one episode.

    Three things the raw zarr will bite you on:
      1. R2, not S3 — s3fs needs the Cloudflare endpoint and region "auto".
      2. obs_keypoints is stored FLAT as (T, 63), not (T, 21, 3).
      3. Arrays are chunk-padded past the end of the episode with ZEROS.
         zarr.json's `total_frames` is authoritative (per the repo changelog);
         the pad rows are a metres-per-frame jump that would fake an impulse
         in literally every episode.
    """
    import os
    import numpy as np
    import s3fs
    import zarr

    endpoint = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
    fs = s3fs.S3FileSystem(
        key=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": endpoint, "region_name": "auto"},
    )
    root = zarr.open(
        zarr.storage.FsspecStore(fs, path=zarr_path.replace("s3://", "").rstrip("/")),
        mode="r",
    )

    key = WRIST_KEY if WRIST_KEY in root else LEFT_KEY  # human_left_arm has left.* only
    arr = np.asarray(root[key])

    if arr.ndim == 2 and arr.shape[1] > 3 and arr.shape[1] % 3 == 0:
        arr = arr.reshape(len(arr), -1, 3)   # (T, 63) -> (T, 21, 3)
    if arr.ndim == 3:
        arr = arr[:, WRIST_JOINT, :]

    n = int(root.attrs.get("total_frames", len(arr)))
    return arr[:n]


# timeout: episode lengths run from 4s to 923s of video, and R2 download
# dominates compute — 300s was a guess. 900 costs nothing unless a container
# actually needs it; measure the real distribution on the --limit 200 run.
@app.function(secrets=[secret], timeout=900, retries=1, max_containers=100)
def audit_one(row: dict) -> dict:
    from eyekit import score_episode
    try:
        xyz = load_wrist(row["zarr_path"])
        rep = score_episode(str(row["episode_id"]), xyz,
                            fps=float(row.get("fps", 30.0)))
        out = rep.to_dict(); out["error"] = ""
        return out
    except Exception as e:                    # never let one episode kill the run
        return {"episode_id": str(row.get("episode_id")), "error": repr(e)}


# Dish family = every task whose name mentions dishes, plus the bare-vessel
# washing tasks. Scoping to the literal string "wash_dishes" gives 42k episodes
# but they are 100% lab=microagi, which kills the by-data-source breakdown.
DISH_FAMILY = r"dish|wash_pot|wash_pan|wash_glass|wash_frying_pan|wash_kitchen_utensil|wash_the_pot"


@app.local_entrypoint()
def main(episodes: str = "episodes.csv", limit: int = 0,
         out: str = "audit_results.parquet",
         task: str = "", lab: str = "", family: bool = False,
         human_only: bool = True, shuffle: bool = False):
    """
      --task wash_dishes      exact task, case-insensitive (catches `Wash_dishes`)
      --family                the whole dish family, across labs
      --lab microagi          restrict to one data source
      --human-only            drop eva_*/yam_* robot embodiments (default on)
      --shuffle               random sample rather than the first N (use with --limit)
    """
    import pandas as pd, time
    df = pd.read_csv(episodes, dtype={"episode_id": str})
    n0 = len(df)

    tasks = df["task"].astype(str).str.lower()
    if family:
        df = df[tasks.str.contains(DISH_FAMILY, regex=True, na=False)]
    elif task:
        df = df[tasks == task.lower()]
    if lab:
        df = df[df["lab"].astype(str).str.lower() == lab.lower()]
    if human_only:
        df = df[df["embodiment"].astype(str).str.startswith("human")]
    if shuffle:
        df = df.sample(frac=1.0, random_state=0)
    if limit:
        df = df.head(limit)

    rows = df.to_dict("records")
    if not rows:
        print("no episodes matched the filters — nothing to do"); return
    print(f"auditing {len(rows)} of {n0} episodes "
          f"| labs={sorted(df['lab'].astype(str).unique())[:6]} "
          f"| tasks={df['task'].nunique()}")

    t0 = time.time()
    results = list(audit_one.map(rows))
    dt = time.time() - t0

    res = pd.DataFrame(results)
    # Carry the grouping keys back from the manifest rather than out of the
    # container: this also attaches lab/task to the *error* rows, so a failure
    # concentrated in one data source is visible instead of silently dropped.
    res = res.merge(df[["episode_id", "lab", "task", "embodiment", "fps",
                        "n_frames"]].rename(columns={"n_frames": "manifest_frames"}),
                    on="episode_id", how="left")
    res.to_parquet(out)

    ok = res[res["error"] == ""] if "error" in res else res
    print(f"\ndone in {dt/60:.1f} min | {len(ok)}/{len(res)} succeeded "
          f"| {dt/max(len(rows),1):.2f}s per episode wall-clock")
    if len(ok):
        prev = ok["failure_flag"].astype(bool).mean()
        print(f"HEADLINE: estimated failure-demo prevalence = {prev:.1%}")
        print(ok.sort_values("failure_score", ascending=False)
                [["episode_id", "task", "lab", "failure_score", "n_impulses",
                  "rf_small_ratio", "eye_opening"]].head(10).to_string())
    bad = res[res["error"] != ""] if "error" in res else res.iloc[:0]
    if len(bad):
        print(f"\n{len(bad)} errors, most common:")
        print(bad["error"].str.slice(0, 90).value_counts().head(5).to_string())
    print(f"wrote {out}")
