"""Smoke test: synthetic clean vs fumbled episodes with known ground truth.
Run: python test_synthetic.py  -> must print PASS before you trust anything."""
import numpy as np
from eyekit import score_episode

rng = np.random.default_rng(7)
FPS = 30.0

def bell(n, peak):
    t = np.linspace(0, 1, n)
    return peak * np.exp(-((t - 0.5) ** 2) / 0.03)

def make_episode(n_cycles=8, fumble=False):
    """Wrist xyz for repeated reach-place cycles (~2s each + pauses)."""
    pos, cur = [], np.zeros(3)
    for c in range(n_cycles):
        n = int(2.0 * FPS)
        spd = bell(n, peak=1.0 + 0.05 * rng.standard_normal())
        if fumble and c in (3, 5):
            # retry: extra half-amplitude correction bump mid-cycle
            spd += np.roll(bell(n, 0.5), n // 4)
        direction = rng.standard_normal(3); direction /= np.linalg.norm(direction)
        for s in spd:
            cur = cur + direction * s / FPS + 0.001 * rng.standard_normal(3)
            pos.append(cur.copy())
        if fumble and c == 4:
            # drop: sharp impulsive displacement over ~3 frames
            for _ in range(3):
                cur = cur + np.array([0, 0, -0.06]) + 0.01 * rng.standard_normal(3)
                pos.append(cur.copy())
        for _ in range(int(0.5 * FPS)):  # quiet pause between cycles
            cur = cur + 0.0005 * rng.standard_normal(3)
            pos.append(cur.copy())
    return np.array(pos)

clean = [score_episode(f"clean{i}", make_episode(), FPS) for i in range(6)]
fumbled = [score_episode(f"fumb{i}", make_episode(fumble=True), FPS) for i in range(6)]

def show(reps, label):
    print(f"\n{label}:")
    for r in reps:
        print(f"  {r.episode_id}: score={r.failure_score:.2f} flag={r.failure_flag} "
              f"impulses={r.n_impulses} rf_small={r.rf_small_ratio:.2f} "
              f"eye={r.eye_opening if np.isfinite(r.eye_opening) else float('nan'):.2f} "
              f"cycles={r.eye_n_cycles}")

show(clean, "CLEAN (should be low score, flag False)")
show(fumbled, "FUMBLED (should be higher score / flagged)")

cs = np.mean([r.failure_score for r in clean])
fs = np.mean([r.failure_score for r in fumbled])
flags_ok = (sum(r.failure_flag for r in clean) <= 1
            and sum(r.failure_flag for r in fumbled) >= 4)
sep_ok = fs > cs + 0.15
print(f"\nmean score clean={cs:.2f} fumbled={fs:.2f}")
print("PASS" if (sep_ok and flags_ok) else "FAIL — tune thresholds in eyekit before proceeding")
