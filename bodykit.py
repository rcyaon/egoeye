"""
bodykit — the multimodal half of the detector: head/gaze, hand aperture, and
body-frame wrist kinematics.

Why this module exists (read NEXT_DIRECTION.md first): wrist trajectory alone
hit a ceiling. Video validation showed every false positive was an *intentional*
forceful motion — soap squirt, garbage-disposal scrape, hard set-down — and an
impulse-magnitude detector cannot separate those from a drop. The discriminator
has to come from channels the wrist doesn't carry:

  * the person LOOKS at what they dropped   -> obs_head_pose
  * the grasp RELEASES                      -> the other 20 hand keypoints
  * the impulse is in the HAND, not the body -> hand measured relative to head

eyekit.py stays frozen (its thresholds were tuned once and defended); this module
adds channels and a gate on top of it.

Everything here is deterministic: no training, no GPU, no learned model.

--------------------------------------------------------------------------
Data conventions — all four VERIFIED on real episodes, not assumed
--------------------------------------------------------------------------
1. `*.obs_keypoints` is flat (T, 63) and reshapes ROW-MAJOR to (T, 21, 3).
   Checked against the alternative (3, 21) transpose: row-major gives a
   15 cm hand (median max joint-to-wrist distance 0.148 m), the transpose
   gives a 1.6 m "hand". Row-major is right.
2. Quaternions in `obs_head_pose` / `obs_wrist_pose` are **wxyz** (scalar
   first). Checked by rebuilding the hand frame from the keypoints and
   comparing: wxyz reproduces the stored rotation to 0.00 deg across every
   frame, xyzw drifts 17-35 deg.
3. Keypoints, wrist/ee pose and head pose all live in ONE shared world frame
   (keypoint joint 0 equals `obs_wrist_pose[:, :3]` to 0.00000 m). So the head
   and the hands are directly comparable — that is what makes gaze geometry
   and the body-frame transform possible.
4. The hand is **MediaPipe-ordered, not MANO-ordered**: joints 1-4 are the
   THUMB (the chain whose base sits 0.040 m from the wrist, against 0.083-
   0.097 m for the finger MCPs), then index 5-8, middle 9-12, ring 13-16,
   pinky 17-20. Verified by bone-length rigidity: 18/20 bones have
   coefficient of variation 0.0000 under this chain. The repo's older
   comments say "MANO"; for joint 0 alone it made no difference, for
   aperture it does.
5. The head camera (`images.front_1`, 640x360, ~105 deg horizontal FOV) shares
   the head-pose frame with NO axis remap: optical x-right / y-down /
   z-forward. Verified by projecting hand keypoints into the JPEG — they land
   on the hand.
"""

from dataclasses import dataclass, field, asdict
import numpy as np

from eyekit import clean_trajectory, kinematics, smooth, detect_impulses, impulse_z

# ----------------------------------------------------------------------
# Hand topology (MediaPipe order — see convention 4 above)
# ----------------------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
FINGER_TIPS = (8, 12, 16, 20)          # index, middle, ring, pinky
MIDDLE_MCP = 9                          # rigid bone 0->9, used as the hand-size scale
PARENTS = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


def unflatten_keypoints(arr: np.ndarray) -> np.ndarray:
    """(T, 63) -> (T, 21, 3), row-major. Passes (T, 21, 3) through."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 63:
        return arr.reshape(len(arr), 21, 3)
    raise ValueError(f"expected (T,63) or (T,21,3) keypoints, got {arr.shape}")


# ----------------------------------------------------------------------
# 1. Rotations
# ----------------------------------------------------------------------

def quat_to_R(q: np.ndarray) -> np.ndarray:
    """(T, 4) wxyz quaternions -> (T, 3, 3) rotation matrices (frame -> world).

    Scalar-first. See convention 2 — this was measured, not guessed.
    """
    q = np.asarray(q, dtype=float)
    w, x, y, z = q.T
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def head_angular_velocity(head_pose: np.ndarray, fps: float):
    """Head angular speed in deg/s, plus its signed pitch (look-down) component.

    Angle of the relative rotation between consecutive frames — the proper way
    to difference orientations; differencing raw quaternion components instead
    would break at the sign flip every quaternion track contains.

    Returns (speed_deg_s, pitch_down_deg_s). Positive pitch = gaze rotating
    DOWNWARD, the direction a dropped object goes — verified, not assumed:
    across 150 episodes this signal correlates with the rate of change of the
    camera forward axis's height at median r = -0.88 (world +z is up here; the
    head's vertical position varies least of the three axes, 0.09 m vs 0.26 m).
    """
    R = quat_to_R(np.asarray(head_pose, dtype=float)[:, 3:7])
    rel = np.einsum("tij,tik->tjk", R[:-1], R[1:])        # R_prev^T @ R_next
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos)) * fps
    # rotation axis in the head frame; x is the pitch axis (y is down, z fwd),
    # so the x-component of the log map is the look-down rate.
    pitch = np.degrees(0.5 * (rel[:, 2, 1] - rel[:, 1, 2])) * fps
    return np.r_[ang[:1], ang], np.r_[pitch[:1], pitch]


# ----------------------------------------------------------------------
# 2. Body frame — the "is the hand moving, or is the person?" fix
# ----------------------------------------------------------------------

def to_body_frame(points_world: np.ndarray, head_pose: np.ndarray) -> np.ndarray:
    """Express world points in the head/body frame: R_head^T (p - head_xyz).

    A wrist impulse in WORLD coordinates fires when the person turns, steps or
    leans — the hand is dragged along by the body and the accelerometer cannot
    tell the difference. Subtracting the head removes that whole confound and
    leaves hand motion relative to the person, which is what "did they fumble"
    actually means.

    points_world: (T, 3) or (T, J, 3). Returns the same shape.
    """
    p = np.asarray(points_world, dtype=float)
    head = np.asarray(head_pose, dtype=float)
    R = quat_to_R(head[:, 3:7])
    single = (p.ndim == 2)
    if single:
        p = p[:, None, :]
    n = min(len(p), len(R))
    rel = p[:n] - head[:n, None, :3]
    out = np.einsum("tij,tkj->tki", R[:n].transpose(0, 2, 1), rel)
    return out[:, 0, :] if single else out


# ----------------------------------------------------------------------
# 3. Hand aperture — the release signature
# ----------------------------------------------------------------------

def hand_aperture(kp: np.ndarray) -> np.ndarray:
    """Grasp aperture per frame: mean thumb-tip-to-fingertip distance, divided
    by the hand's own rigid wrist->middle-MCP bone so it is scale-free across
    people (that bone's length varies 0.00% within an episode — see convention 4).

    A grasp release is a step increase; a squeeze is a decrease. This is the
    channel that separates "let go of the plate" from "pushed something hard".
    """
    kp = unflatten_keypoints(kp)
    tips = kp[:, list(FINGER_TIPS), :]                       # (T, 4, 3)
    d = np.linalg.norm(tips - kp[:, [THUMB_TIP], :], axis=2)  # (T, 4)
    scale = np.median(np.linalg.norm(kp[:, MIDDLE_MCP] - kp[:, WRIST], axis=1))
    return d.mean(axis=1) / (scale + 1e-9)


def aperture_opening_rate(aperture: np.ndarray, fps: float) -> np.ndarray:
    """d(aperture)/dt, smoothed. Positive = hand opening = releasing."""
    a = smooth(np.nan_to_num(aperture, nan=float(np.nanmedian(aperture))), fps, win_s=0.17)
    return np.gradient(a, 1.0 / fps)


# ----------------------------------------------------------------------
# 4. Gaze geometry — where the hand sits in the head camera's view
# ----------------------------------------------------------------------

def project_to_camera(points_world: np.ndarray, head_pose: np.ndarray,
                      K: np.ndarray, image_wh=(640, 360)):
    """Project world points into the head camera. Returns (u, v, z, in_view).

    This is the "take the pixel data into account" step: the hand's position in
    the operator's own field of view, in the units the camera actually measures.
    No axis remap — the optical frame IS the head-pose frame (convention 5).
    """
    K = np.asarray(K, dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam = to_body_frame(points_world, head_pose)
    if cam.ndim == 3:
        cam = cam.reshape(len(cam), -1, 3)
    z = cam[..., 2]
    zz = np.where(z > 1e-6, z, np.nan)
    u = fx * cam[..., 0] / zz + cx
    v = fy * cam[..., 1] / zz + cy
    W, H = image_wh
    in_view = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, in_view


def gaze_hand_angle(hand_world: np.ndarray, head_pose: np.ndarray) -> np.ndarray:
    """Angle (deg) between where the head points and where the hand is.

    0 = looking straight at the hand. Rises when attention leaves the hand.
    Computed from geometry rather than pixels so it stays defined when the hand
    leaves the frame — but it is the same quantity the projection measures.
    """
    cam = to_body_frame(hand_world, head_pose)          # (T, 3) in head frame
    d = np.linalg.norm(cam, axis=1) + 1e-9
    return np.degrees(np.arccos(np.clip(cam[:, 2] / d, -1.0, 1.0)))


# ----------------------------------------------------------------------
# 5. Robust event detection, shared shape with eyekit.detect_impulses
# ----------------------------------------------------------------------

def _robust_z(x: np.ndarray, fps: float, hp_win_s: float = 0.33) -> np.ndarray:
    """High-pass then MAD-z-score. Same playbook as eyekit.impulse_z: a spike
    must not be allowed to set its own scale, so the scale is median-based."""
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else 0.0)
    resid = x - smooth(x, fps, win_s=hp_win_s)
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid))) + 1e-9
    return resid / (1.4826 * mad)          # SIGNED: direction matters here


def detect_events(signal: np.ndarray, fps: float, z_thresh: float,
                  min_sep_s: float = 0.3, signed: bool = True):
    """Frame indices where `signal` spikes, grouped so one physical event is one
    detection. `signed=True` keeps only upward spikes (hand OPENING, gaze
    accelerating), which is the direction that carries meaning."""
    z = _robust_z(signal, fps)
    zz = z if signed else np.abs(z)
    hits = np.where(zz > z_thresh)[0]
    events = []
    for h in hits:
        if not events or h - events[-1][-1] > min_sep_s * fps:
            events.append([h])
        else:
            events[-1].append(h)
    return np.array([int(np.mean(e)) for e in events], dtype=int), zz


def coincidences(a: np.ndarray, b: np.ndarray, fps: float, window_s: float = 1.0):
    """Indices of events in `a` that have any event of `b` within +/- window_s."""
    if len(a) == 0 or len(b) == 0:
        return np.array([], dtype=int)
    w = window_s * fps
    return np.array([i for i, t in enumerate(a) if np.min(np.abs(b - t)) <= w], dtype=int)


def coincidence_lift(a: np.ndarray, b: np.ndarray, n_frames: int, fps: float,
                     window_s: float = 1.0) -> float:
    """Observed coincidence count / count expected if the two channels were
    independent. This is the statistic that makes the multimodal claim testable
    WITHOUT video labels.

    If `b` were sprinkled at random over the episode, each `a` event would catch
    one with probability ~ 1 - (1 - 2*w/T)^|b|. Lift > 1 means wrist impulses and
    gaze/release events genuinely co-occur; lift ~ 1 means we are just ANDing two
    noisy channels and should not claim a discriminator.
    """
    if len(a) == 0 or len(b) == 0 or n_frames <= 0:
        return np.nan
    w = window_s * fps
    p = 1.0 - max(0.0, 1.0 - 2.0 * w / n_frames) ** len(b)
    expected = p * len(a)
    if expected <= 0:
        return np.nan
    return float(len(coincidences(a, b, fps, window_s)) / expected)


# ----------------------------------------------------------------------
# 6. Episode scoring
# ----------------------------------------------------------------------

# --------------------------------------------------------------------------
# Operating point. Chosen against a CIRCULAR-SHIFT null: shifting a supporting
# channel by a random offset inside its own episode preserves that channel's
# event count and burst structure exactly and destroys only its alignment to the
# impulses, so
#       enrichment = confirmations(real) / confirmations(shifted)
# measures co-timing rather than "both channels are busy at once". A uniform-
# random null cannot do that — it under-counts coincidences for bursty channels
# and reports lift where there is none.
#
# Swept over 133 min of real wash_dishes video (400 episodes, 800 hands):
#
#   rule                        window   confirmed   null   enrichment   flag rate
#   OR  (z_head 5, z_rel 5)      1.0 s      117      61.4      1.9x        16.8%
#   OR  (z_head 5, z_rel 5)      0.5 s      103      39.8      2.6x        15.8%
#   AND (z_head 5, z_rel 5)      1.0 s       44       9.3      4.7x         7.2%
#   AND (z_head 5, z_rel 5)      0.5 s       22       3.1      7.1x         4.0%
#
# Two things this settles. NEXT_DIRECTION.md proposed OR ("impulse AND (head OR
# hand-open)"); AND is three times more selective, and it is also the right
# physics — letting go of a plate opens the hand *and* pulls the gaze. And the
# co-timing is tight: half a second beats a full second everywhere, which
# matches the lag profile, whose peak decays inside ~0.5 s.
#
# z=5 on both channels sits on a plateau rather than a spike (its neighbours
# score 6.0-7.5x), so it is not a knife-edge fit to this sample.
#
# That sweep predates the edge guard below. With edge_s applied the same
# operating point gives 14 confirmed against a 2.4 null — 5.96x, 3.0% flag rate
# — and it holds on data it was never tuned on (fold_clothes, 6.45x). The guard
# removes about a third of the impulses and leaves the enrichment intact.
# --------------------------------------------------------------------------
DEFAULTS = {
    "z_impulse": 10.0,      # inherited unchanged from eyekit
    "z_head": 5.0,          # head angular-velocity / pitch spike
    "z_release": 5.0,       # aperture opening-rate spike (upward only)
    "window_s": 0.5,        # supporting evidence must be this close in time
    "require_both": True,   # AND, not OR — see the table above
    "body_veto_s": 0.3,     # impulse must also show up in the body frame
    "edge_s": 1.0,          # ignore events this close to either end of the clip
    "min_confirmed": 1,     # >=1 confirmed impulse flags the episode
}

# edge_s is not cosmetic. Rendering the first calls as filmstrips showed two of
# six sitting at 0.7-0.9 s — the clip's own beginning, where every one of these
# channels is untrustworthy at once: savgol smoothing is ill-conditioned at a
# boundary, np.gradient falls back to one-sided differences, and the recording
# itself typically opens mid-motion with the camera still settling (those frames
# are visibly motion-blurred). Episodes here have a 12 s median, so a 1 s guard
# at each end is cheap next to the false calls it removes.

# --------------------------------------------------------------------------
# Which channels may be ANDed together — this is a correctness constraint,
# not a style preference.
#
# The gate's whole claim is that two INDEPENDENT observations coincide. That
# claim is void if the two channels are computed from overlapping inputs, and
# it is easy to do by accident here:
#
#   * body-frame wrist acceleration is built FROM the head rotation, so pairing
#     it with a head channel partly correlates head motion with itself;
#   * gaze-hand angle is built from the head AND the hand, so pairing it with
#     any wrist channel does the same.
#
# Measured, on 229 min of real wash_dishes video, as the ratio of the
# coincidence rate at lag 0 to its rate at |lag| > 2 s:
#
#     impulse(world) x head angular speed   2.95   <- no shared inputs
#     impulse(world) x head pitch-down      3.83   <- no shared inputs
#     impulse(world) x aperture rate        3.71   <- no shared inputs
#     impulse(body)  x head angular speed   2.86
#     impulse(body)  x gaze-hand angle rate 6.27   <- inflated by shared inputs
#
# So the gate pairs the WORLD-frame impulse (hand keypoints only) with head
# channels (head pose only) and aperture (hand keypoints, other joints). The
# body frame is still used, but as a VETO — "was this the hand or the whole
# person moving" — which consumes no coincidence budget. gaze_hand_angle stays
# in the report as a descriptive statistic and is deliberately not in the gate.
# --------------------------------------------------------------------------


@dataclass
class MultimodalReport:
    episode_id: str
    hand: str = ""
    n_frames: int = 0
    duration_s: float = np.nan
    fps: float = np.nan

    # channel event counts
    n_impulses_world: int = 0        # what the old wrist-only detector saw
    n_impulses_body: int = 0         # same detector run in the head/body frame
    n_impulses_hand: int = 0         # world impulses that survive the body veto
    n_head_events: int = 0
    n_release_events: int = 0

    # the gate
    n_confirmed: int = 0             # impulses with gaze OR release support
    n_confirmed_head: int = 0
    n_confirmed_release: int = 0
    confirmed_frames: list = field(default_factory=list)
    confirmed_per_min: float = np.nan

    # chance baselines — the honesty check
    lift_head: float = np.nan
    lift_release: float = np.nan

    # descriptive channel stats
    aperture_median: float = np.nan
    aperture_p95: float = np.nan
    head_angspeed_median: float = np.nan
    gaze_hand_angle_median: float = np.nan
    frac_hand_in_view: float = np.nan
    body_motion_frac: float = np.nan   # share of world impulses that were the BODY

    failure_score: float = np.nan
    failure_flag: bool = False

    def to_dict(self):
        d = asdict(self)
        d["confirmed_frames"] = list(map(int, d["confirmed_frames"]))
        return d


def score_episode_multimodal(episode_id: str, kp: np.ndarray, head_pose: np.ndarray,
                             fps: float, hand: str = "", K=None,
                             image_wh=(640, 360),
                             thresholds: dict | None = None) -> MultimodalReport:
    """Score one hand of one episode on all channels + the multimodal gate.

    kp:        (T, 63) or (T, 21, 3) hand keypoints, world frame
    head_pose: (T, 7) xyz + wxyz quaternion, world frame
    """
    th = {**DEFAULTS, **(thresholds or {})}
    kp = unflatten_keypoints(kp)
    head_pose = np.asarray(head_pose, dtype=float)
    n = min(len(kp), len(head_pose))
    kp, head_pose = kp[:n], head_pose[:n]

    rep = MultimodalReport(episode_id=episode_id, hand=hand, n_frames=n, fps=float(fps))
    rep.duration_s = float(n / max(float(fps), 1e-9))
    if n < 16:
        return rep

    wrist_world = clean_trajectory(kp[:, WRIST, :])
    wrist_body = to_body_frame(wrist_world, head_pose)

    # --- channel 1: wrist impulses. World frame is the gate's channel (it
    # touches no head data); the body frame is the veto. -------------------
    _, accel_w = kinematics(wrist_world, fps)
    imp_world, _ = detect_impulses(accel_w, fps, z_thresh=th["z_impulse"])
    _, accel_b = kinematics(wrist_body, fps)
    imp_body, _ = detect_impulses(accel_b, fps, z_thresh=th["z_impulse"])
    rep.n_impulses_world, rep.n_impulses_body = len(imp_world), len(imp_body)

    # Veto: an impulse that vanishes once the head's own motion is removed was
    # the person moving, not the hand. On real wash_dishes data this discards
    # roughly half of all raw impulses.
    keep = coincidences(imp_world, imp_body, fps, window_s=th["body_veto_s"])
    imp = imp_world[keep] if len(imp_world) else imp_world
    if len(imp_world):
        rep.body_motion_frac = float(1.0 - len(keep) / len(imp_world))
    # edge guard — see the note on edge_s
    edge = int(th["edge_s"] * fps)
    if len(imp):
        imp = imp[(imp >= edge) & (imp < n - edge)]
    rep.n_impulses_hand = int(len(imp))

    # --- channel 2: head / gaze (head pose only) ---------------------------
    angspeed, pitch = head_angular_velocity(head_pose, fps)
    ev_speed, _ = detect_events(angspeed, fps, z_thresh=th["z_head"], signed=True)
    ev_pitch, _ = detect_events(pitch, fps, z_thresh=th["z_head"], signed=True)
    head_ev = np.union1d(ev_speed, ev_pitch).astype(int)
    rep.n_head_events = len(head_ev)
    rep.head_angspeed_median = float(np.nanmedian(angspeed))

    # --- channel 3: hand aperture (hand keypoints, non-wrist joints) -------
    ap = hand_aperture(kp)
    rate = aperture_opening_rate(ap, fps)
    rel_ev, _ = detect_events(rate, fps, z_thresh=th["z_release"], signed=True)
    rep.n_release_events = len(rel_ev)
    rep.aperture_median = float(np.nanmedian(ap))
    rep.aperture_p95 = float(np.nanpercentile(ap, 95))

    # --- descriptive only: mixes hand and head, so it must stay out of the gate
    rep.gaze_hand_angle_median = float(np.nanmedian(gaze_hand_angle(wrist_world, head_pose)))
    if K is not None:
        _, _, _, in_view = project_to_camera(kp, head_pose, K, image_wh)
        rep.frac_hand_in_view = float(np.nanmean(in_view))

    # --- the gate ---------------------------------------------------------
    i_head = coincidences(imp, head_ev, fps, th["window_s"])
    i_rel = coincidences(imp, rel_ev, fps, th["window_s"])
    confirmed = (np.intersect1d(i_head, i_rel) if th["require_both"]
                 else np.union1d(i_head, i_rel)).astype(int)
    rep.n_confirmed = int(len(confirmed))
    rep.n_confirmed_head, rep.n_confirmed_release = int(len(i_head)), int(len(i_rel))
    rep.confirmed_frames = [int(imp[i]) for i in confirmed]
    rep.confirmed_per_min = float(rep.n_confirmed / max(rep.duration_s / 60.0, 1e-9))

    rep.lift_head = coincidence_lift(imp, head_ev, n, fps, th["window_s"])
    rep.lift_release = coincidence_lift(imp, rel_ev, n, fps, th["window_s"])

    # Score by how much independent evidence lines up, NOT by how hard the
    # impulse was — magnitude is precisely what failed video validation
    # (p@10 = 1/10: the hardest impulses were soap squirts and disposal
    # scrapes). A single confirmed event carries most of the score; a second
    # one adds the rest, because repeated confirmed events in one short clip
    # is what a genuine drop-and-recover looks like.
    rep.failure_score = float(np.clip(0.7 * min(rep.n_confirmed, 1)
                                      + 0.3 * min(max(rep.n_confirmed - 1, 0), 1), 0, 1))
    rep.failure_flag = bool(rep.n_confirmed >= int(th["min_confirmed"]))
    return rep
