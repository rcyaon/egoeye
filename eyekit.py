"""
eyekit — signal-integrity-style analysis of human demonstration kinematics.

Pipeline: wrist xyz -> speed/accel -> (a) impulsive-event detection
(kurtosis / crest factor), (b) rainflow cycle decomposition,
(c) motion-cycle segmentation -> eye diagram -> eye-opening score.

Everything is deterministic. No training, no GPU.
"""

from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.signal import savgol_filter, find_peaks

import rainflow

# ----------------------------------------------------------------------
# 1. Preprocessing
# ----------------------------------------------------------------------

def clean_trajectory(xyz: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Interpolate short NaN gaps in an (T, 3) trajectory; long gaps stay NaN.

    Decide max_gap NOW (frames), not at 3pm. Segments containing long gaps
    are excluded downstream rather than hallucinated.
    """
    xyz = np.asarray(xyz, dtype=float).copy()
    t = np.arange(len(xyz))
    for d in range(xyz.shape[1]):
        col = xyz[:, d]
        nans = np.isnan(col)
        if not nans.any():
            continue
        # find NaN runs
        runs = _runs(nans)
        for start, stop in runs:
            if (stop - start) <= max_gap and start > 0 and stop < len(col):
                col[start:stop] = np.interp(
                    t[start:stop], [start - 1, stop], [col[start - 1], col[stop]]
                )
        xyz[:, d] = col
    return xyz


def _runs(mask: np.ndarray):
    """[(start, stop), ...] for True-runs in a boolean mask (stop exclusive)."""
    out, in_run, start = [], False, 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            out.append((start, i)); in_run = False
    if in_run:
        out.append((start, len(mask)))
    return out


def kinematics(xyz: np.ndarray, fps: float):
    """Return (speed, accel_mag) 1D arrays from (T,3) positions."""
    vel = np.gradient(xyz, 1.0 / fps, axis=0)          # (T,3) m/s
    speed = np.linalg.norm(vel, axis=1)                # (T,)
    acc = np.gradient(vel, 1.0 / fps, axis=0)          # (T,3) m/s^2
    accel_mag = np.linalg.norm(acc, axis=1)
    return speed, accel_mag


def smooth(x: np.ndarray, fps: float, win_s: float = 0.25):
    w = max(5, int(win_s * fps) | 1)  # odd, >=5
    if len(x) <= w:
        return x
    return savgol_filter(x, w, polyorder=2)

# ----------------------------------------------------------------------
# 2. Impulsive events (drops / collisions): sliding kurtosis + crest factor
#    (envelope-analysis playbook from rotating-machinery diagnostics)
# ----------------------------------------------------------------------

def impulse_z(accel_mag: np.ndarray, fps: float, hp_win_s: float = 0.33):
    """Robust z-score of the high-passed acceleration.

    Why not window-RMS crest factor: a spike inflates its own window's RMS
    and hides itself (verified on synthetic data). Instead: subtract the
    smooth low-frequency component (intentional motion is smooth; impacts
    are broadband), then z-score the residual against MAD — the impact
    can't inflate a median-based scale.
    """
    x = np.nan_to_num(accel_mag, nan=np.nanmedian(accel_mag))
    resid = x - smooth(x, fps, win_s=hp_win_s)
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid))) + 1e-9
    return np.abs(resid) / (1.4826 * mad)


def detect_impulses(accel_mag: np.ndarray, fps: float, z_thresh: float = 10.0,
                    min_sep_s: float = 0.3):
    """Frame indices of impulsive events (drops/collisions), grouped so one
    physical event = one detection. Threshold has huge margin on synthetic
    data (clean z_max ~5, drop z_max ~35+); verify it holds on real data."""
    z = impulse_z(accel_mag, fps)
    hits = np.where(z > z_thresh)[0]
    events = []
    for h in hits:
        if not events or h - events[-1][-1] > min_sep_s * fps:
            events.append([h])
        else:
            events[-1].append(h)
    return np.array([int(np.mean(e)) for e in events], dtype=int), z

# ----------------------------------------------------------------------
# 3. Rainflow: correction/retry signature = swarm of small cycles
# ----------------------------------------------------------------------

def rainflow_features(speed: np.ndarray, noise_frac: float = 0.1,
                      small_hi: float = 0.5):
    """Ratio of small-to-total *real* motion cycles.

    Three bands (as fractions of max cycle range):
      < noise_frac        -> tracking noise, IGNORED (on synthetic data these
                             are ~90% of all rainflow cycles and saturate any
                             naive ratio — verified failure mode)
      [noise_frac, small_hi) -> corrections / re-grasps / dithering
      >= small_hi         -> intentional motions
    """
    x = speed[~np.isnan(speed)]
    if len(x) < 10:
        return {"rf_n_cycles": 0, "rf_small_ratio": np.nan}
    cyc = list(rainflow.count_cycles(x))
    if not cyc:
        return {"rf_n_cycles": 0, "rf_small_ratio": np.nan}
    rngs = np.array([r for r, _ in cyc], dtype=float)
    cnts = np.array([c for _, c in cyc], dtype=float)
    mx = rngs.max()
    small = cnts[(rngs >= noise_frac * mx) & (rngs < small_hi * mx)].sum()
    large = cnts[rngs >= small_hi * mx].sum()
    total = small + large
    return {
        "rf_n_cycles": float(total),
        "rf_small_ratio": float(small / total) if total else np.nan,
    }

# ----------------------------------------------------------------------
# 4. Motion cycles + eye diagram
# ----------------------------------------------------------------------

def segment_cycles(speed: np.ndarray, fps: float,
                   min_cycle_s: float = 0.6, quiet_frac: float = 0.15):
    """Split a speed trace into motion cycles at quiet valleys.

    Deliberately dumb: smooth, find peaks with prominence & spacing,
    cut at the minimum between consecutive peaks. Skip DTW. Skip learning.
    Returns list of (start, stop) frame index pairs.
    """
    sm = smooth(np.nan_to_num(speed, nan=0.0), fps)
    if sm.max() <= 0:
        return []
    peaks, _ = find_peaks(
        sm,
        prominence=0.3 * sm.max(),
        distance=int(min_cycle_s * fps),
        height=quiet_frac * sm.max(),
    )
    if len(peaks) < 2:
        return []
    bounds = [int(np.argmin(sm[a:b]) + a) for a, b in zip(peaks[:-1], peaks[1:])]
    bounds = [0] + bounds + [len(sm) - 1]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (b - a) < int(min_cycle_s * fps):
            continue
        # trim to active region: variable-length pauses inside segments
        # smear the eye with fake phase noise (verified on synthetic data)
        seg = sm[a:b]
        active = np.where(seg > 0.2 * seg.max())[0]
        if len(active) < int(0.3 * fps):
            continue
        segs.append((a + int(active[0]), a + int(active[-1]) + 1))
    return segs


def eye_matrix(speed: np.ndarray, segs, n: int = 100, max_nan_frac: float = 0.2):
    """Resample each cycle to n samples; stack -> (n_cycles, n). Peak-normalized
    per-episode (not per-cycle: amplitude excursions should smear the eye)."""
    rows = []
    for a, b in segs:
        seg = speed[a:b]
        if np.isnan(seg).mean() > max_nan_frac or len(seg) < 4:
            continue
        seg = np.nan_to_num(seg, nan=0.0)
        rows.append(np.interp(np.linspace(0, len(seg) - 1, n),
                              np.arange(len(seg)), seg))
    if not rows:
        return None
    M = np.vstack(rows)
    scale = np.percentile(M, 99) + 1e-9
    return M / scale


def eye_metrics(M: np.ndarray):
    """Eye-opening score in [0,1]: median profile energy vs. phase-wise spread.
    High = tight eye = consistent behavior. Also returns mask-violation
    fraction per cycle (p5–p95 envelope)."""
    med = np.median(M, axis=0)
    iqr = np.percentile(M, 75, axis=0) - np.percentile(M, 25, axis=0)
    act = med > 0.25 * med.max()          # spread only where motion happens;
    if act.sum() < 5:                      # near-zero median phases otherwise
        act = np.ones_like(act, bool)      # blow up the ratio arbitrarily
    opening = float(np.clip(1.0 - iqr[act].mean() / (med[act].mean() + 1e-9), 0, 1))
    lo, hi = np.percentile(M, 5, axis=0), np.percentile(M, 95, axis=0)
    viol = ((M < lo) | (M > hi)).mean(axis=1)  # per-cycle fraction outside mask
    return {
        "eye_opening": opening,
        "eye_n_cycles": int(M.shape[0]),
        "mask_violation_p90": float(np.percentile(viol, 90)),
    }

# ----------------------------------------------------------------------
# 5. Episode-level scoring
# ----------------------------------------------------------------------

@dataclass
class EpisodeReport:
    episode_id: str
    n_frames: int = 0
    duration_s: float = np.nan
    nan_frac: float = np.nan
    n_impulses: int = 0
    impulse_rate_per_min: float = np.nan
    impulse_frames: list = field(default_factory=list)
    rf_small_ratio: float = np.nan
    rf_n_cycles: float = np.nan
    eye_opening: float = np.nan
    eye_n_cycles: int = 0
    mask_violation_p90: float = np.nan
    failure_flag: bool = False
    failure_score: float = np.nan  # 0 clean .. 1 confident failure
    def to_dict(self):
        d = asdict(self); d["impulse_frames"] = list(map(int, d["impulse_frames"]))
        return d


def score_episode(episode_id: str, wrist_xyz: np.ndarray, fps: float,
                  thresholds: dict | None = None) -> EpisodeReport:
    """Full deterministic pipeline for one episode. Tune thresholds ONCE on a
    dev handful, then freeze — that's your defensibility story."""
    th = {
        "z": 10.0,
        "eye_open_min": 0.45,
        "impulses_per_min_min": 1.2,
        "min_impulses": 2,
        "fail_score_min": 0.55,
        **(thresholds or {}),
    }
    rep = EpisodeReport(episode_id=episode_id, n_frames=len(wrist_xyz))
    rep.duration_s = float(rep.n_frames / max(float(fps), 1e-9))
    xyz = clean_trajectory(wrist_xyz)
    rep.nan_frac = float(np.isnan(xyz).any(axis=1).mean())
    speed, accel = kinematics(xyz, fps)

    imp, _ = detect_impulses(accel, fps, z_thresh=th["z"])
    rep.n_impulses, rep.impulse_frames = len(imp), imp.tolist()
    dur_min = max(rep.duration_s / 60.0, 1e-9)
    rep.impulse_rate_per_min = float(rep.n_impulses / dur_min)

    rf = rainflow_features(speed)
    rep.rf_small_ratio, rep.rf_n_cycles = rf["rf_small_ratio"], rf["rf_n_cycles"]

    segs = segment_cycles(speed, fps)
    M = eye_matrix(speed, segs)
    if M is not None and M.shape[0] >= 3:
        rep.__dict__.update(eye_metrics(M))

    # --- combine (weighted: impulses are the reliable channel, rainflow
    # secondary, eye smear tertiary — margins measured on synthetic data) ---
    w, s = [], []
    w.append(0.6); s.append(min(rep.n_impulses / 1.0, 1.0))            # drops
    if np.isfinite(rep.rf_small_ratio):
        w.append(0.25)
        s.append(np.clip((rep.rf_small_ratio - 0.05) / 0.3, 0, 1))     # retries
    if np.isfinite(rep.eye_opening):
        w.append(0.15)
        s.append(np.clip((th["eye_open_min"] - rep.eye_opening)
                         / th["eye_open_min"], 0, 1))                  # smear
    rep.failure_score = float(np.dot(w, s) / np.sum(w)) if w else np.nan

    impulse_gate = (
        rep.n_impulses >= int(th["min_impulses"])
        and rep.impulse_rate_per_min >= float(th["impulses_per_min_min"])
    )
    rep.failure_flag = bool(impulse_gate or rep.failure_score >= float(th["fail_score_min"]))
    return rep


def confidence_trace(wrist_xyz: np.ndarray, fps: float, z_thresh: float = 10.0):
    """Per-frame 'success confidence' in [0,1] for the demo confidence meter.
    Confidence dips exactly at impulsive events (z-score saturates at 2x
    threshold), with a short smear so dips are visible at video framerates."""
    xyz = clean_trajectory(wrist_xyz)
    _, accel = kinematics(xyz, fps)
    _, z = detect_impulses(accel, fps, z_thresh=z_thresh)
    risk = np.clip((z - 0.5 * z_thresh) / (1.5 * z_thresh), 0, 1)
    k = max(3, int(0.25 * fps))
    risk = np.convolve(risk, np.ones(k) / k, mode="same") * k  # widen dips
    risk = np.clip(risk, 0, 1)
    t = np.arange(len(xyz)) / fps
    return t, 1.0 - risk
