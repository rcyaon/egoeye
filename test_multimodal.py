"""
test_multimodal.py — ground-truth smoke test for the multimodal gate.

    python test_multimodal.py     ->  PASS / FAIL

Same idea as test_synthetic.py: build episodes whose answer we know and check
the detector agrees. The four cases are exactly the confusions this gate was
built to resolve, so if any of them regress the gate has stopped doing its job:

  clean   smooth motion, steady grasp, steady head            -> no flag
  drop    impulse + hand opens + gaze snaps, all co-timed     -> FLAG
  squirt  identical impulse, grasp and gaze unchanged         -> no flag
          (this is the soap-squirt/disposal-scrape false positive that put the
           wrist-only detector at precision@10 = 1/10)
  body    the head swings and drags the hand with it          -> no flag
          (a world-frame impulse with no hand motion behind it)

Also asserts the conventions bodykit documents, so a future refactor that
transposes the keypoints or swaps the quaternion order fails loudly here.
"""
import numpy as np

import bodykit as bk

FPS = 30.0
T = 300
EVENT = 150


def _hand(aperture_scale):
    """21 keypoints in a canonical hand pose, MediaPipe order (thumb 1-4).

    aperture_scale: (T,) multiplier on the thumb-to-fingertip gap. 1.0 is the
    grasp; larger is an open hand.
    """
    # rest pose relative to the wrist, roughly to scale in metres
    rest = np.zeros((21, 3))
    rest[1:5] = [[0.02, -0.02, 0], [0.035, -0.032, 0], [0.045, -0.04, 0], [0.055, -0.048, 0]]
    for c, (i0, y) in enumerate([(5, 0.030), (9, 0.010), (13, -0.010), (17, -0.030)]):
        for k in range(4):
            rest[i0 + k] = [0.05 + 0.028 * k, y, 0.0]
    kp = np.repeat(rest[None], T, axis=0)
    # opening moves the four fingertips away from the thumb tip
    for tip in bk.FINGER_TIPS:
        kp[:, tip, 1] += (aperture_scale - 1.0) * 0.05
    return kp


def _episode(impulse=False, opens=False, head_turn=False, body_swing=False):
    t = np.arange(T) / FPS
    ap = np.ones(T)
    if opens:
        ap += 1.4 / (1.0 + np.exp(-(np.arange(T) - EVENT) / 1.5))   # step open

    kp = _hand(ap)
    # smooth reaching motion for the wrist, plus per-joint offsets
    wrist = np.stack([0.30 * np.sin(2 * np.pi * 0.25 * t),
                      0.10 * np.cos(2 * np.pi * 0.25 * t),
                      0.90 + 0.05 * np.sin(2 * np.pi * 0.5 * t)], axis=1)
    if impulse:                       # sharp one-frame velocity kick
        kick = np.zeros((T, 3))
        kick[EVENT:] = [0.045, -0.030, -0.035]
        wrist = wrist + kick
    kp = kp + wrist[:, None, :]

    head = np.zeros((T, 7))
    head[:, :3] = [0.0, 0.0, 1.55]
    head[:, 3] = 1.0                  # wxyz identity
    if head_turn:                     # fast pitch-down, settling
        ang = 0.7 / (1.0 + np.exp(-(np.arange(T) - EVENT) / 1.2))
        head[:, 3], head[:, 4] = np.cos(ang / 2), np.sin(ang / 2)
    if body_swing:
        # the head rotates and the hand is carried rigidly with it: a world-frame
        # impulse with nothing happening in the hand
        ang = 0.9 / (1.0 + np.exp(-(np.arange(T) - EVENT) / 1.0))
        head[:, 3], head[:, 4] = np.cos(ang / 2), np.sin(ang / 2)
        R = bk.quat_to_R(head[:, 3:7])
        kp = np.einsum("tij,tkj->tki", R, kp - head[:, None, :3]) + head[:, None, :3]
    return kp, head


def main() -> int:
    fails = []

    def check(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
        if not cond:
            fails.append(name)

    print("conventions")
    kp, head = _episode()
    check("keypoints reshape row-major", bk.unflatten_keypoints(
        kp.reshape(T, 63)).shape == (T, 21, 3))
    R = bk.quat_to_R(np.array([[1.0, 0, 0, 0]]))
    check("quat wxyz identity -> I", np.allclose(R[0], np.eye(3)))
    # a 90 deg rotation about x sends +y to +z under wxyz
    q = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]])
    check("quat wxyz sign convention",
          np.allclose(bk.quat_to_R(q)[0] @ np.array([0, 1, 0]), [0, 0, 1], atol=1e-6))
    ap = bk.hand_aperture(_hand(np.ones(T)))
    ap_open = bk.hand_aperture(_hand(np.full(T, 2.0)))
    check("aperture rises when the hand opens", ap_open.mean() > ap.mean() * 1.3,
          f"{ap.mean():.3f} -> {ap_open.mean():.3f}")
    # A hand held rigidly in front of a swinging head sweeps a long arc in world
    # coordinates and must be perfectly still in the body frame — that is the
    # whole point of the veto. Tested on its own so no independent hand motion
    # can mask it.
    ang = 0.9 / (1.0 + np.exp(-(np.arange(T) - EVENT) / 1.0))
    head_r = np.zeros((T, 7))
    head_r[:, :3] = [0.0, 0.0, 1.55]
    head_r[:, 3], head_r[:, 4] = np.cos(ang / 2), np.sin(ang / 2)
    held = np.array([0.0, 0.0, 0.45])                       # 45 cm in front of the face
    rigid = np.einsum("tij,j->ti", bk.quat_to_R(head_r[:, 3:7]), held) + head_r[:, :3]
    world_ptp = np.ptp(rigid, axis=0).max()
    body_ptp = np.ptp(bk.to_body_frame(rigid, head_r), axis=0).max()
    check("body frame cancels rigid body motion", body_ptp < 1e-6 < world_ptp,
          f"world ptp {world_ptp:.3f} m -> body ptp {body_ptp:.2e} m")

    print("\ndetector")
    cases = {
        "clean  (nothing happens)":        (dict(), False),
        "drop   (impulse+open+gaze)":      (dict(impulse=True, opens=True, head_turn=True), True),
        "squirt (impulse only)":           (dict(impulse=True), False),
        "body   (head swings, hand rides)": (dict(body_swing=True), False),
    }
    for name, (kwargs, want) in cases.items():
        kp, head = _episode(**kwargs)
        r = bk.score_episode_multimodal("synthetic", kp, head, FPS, hand="right")
        got = r.failure_flag
        check(f"{name:<34s} -> {'FLAG' if want else 'no flag'}", got == want,
              f"confirmed={r.n_confirmed} impulses world={r.n_impulses_world} "
              f"hand={r.n_impulses_hand} head={r.n_head_events} rel={r.n_release_events}")

    print("\nPASS" if not fails else f"\nFAIL — {len(fails)}: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
