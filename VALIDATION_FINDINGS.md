# Validation findings — precision@10 on wash_dishes (human-watched)

**Bottom line: the impulse detector flags forceful *intentional* motion, not failure.
precision@10 = 1/10.** We cannot report a "failure prevalence" number as-is. But the
watch showed the real discriminator, so the fix is clear.

## What we ran
- Corpus corrected to **`wash_dishes`** (1,500 eps, 100% microagi, median clip 11s).
  Our earlier cross-task sample was dominated by fold_clothes / cup_on_saucer, so its
  top calls weren't even dishwashing. `audit_washdishes.parquet` (gitignored).
- Per-episode flag rate on wash_dishes: **21.5%**. Human validation shows this is
  over-flagging.

## precision@10 — watched the 10 most-confident failure calls
| # | verdict | what it actually was |
|---|---|---|
| 1–3 | N | normal washing |
| 4 | N | scraping garbage disposal into trash (intentional forceful motion) |
| 5 | N | normal washing |
| 6 | N | normal washing |
| 7 | N | normal washing |
| 8 | N | normal washing |
| **9** | **Y** | **dish placed → fell over → corrected → dish fell in sink → tossed** |
| 10 | N | squirting soap (impulse from the squirt, not a failure) |

**precision@10 = 1/10.** Every false positive is an *intentional* impulsive action —
soap squirt, disposal scrape, hard set-down. Impulse magnitude alone cannot tell these
from a real drop.

## The fix the watch revealed
The one true failure (#9) had an **impulse followed by a correction / re-grab**. The
intentional false-positives do NOT. So the discriminator is the **recovery signature**
(rainflow small-cycle burst *after* an impulse), not impulse magnitude.
→ Person B: gate a failure on `impulse AND a following correction`, re-validate.

## Recommendation
1. **Reframe the claim** (don't report raw prevalence): "deterministic kinematics flags
   forceful motion; validated on video (p@10=10%); real failures need an impulse+correction
   signature." A rigorous negative result + a validated direction — exactly what Track 3's
   "labels are subjective" problem rewards, and it survives a judge watching a clip.
2. Optionally hand B the impulse+correction fix and re-run (higher time risk).

## Artifacts
- Clips watched: `~/Documents/Hackathon/clips_wd/` (+ scorecard)
- wash_dishes scores: `audit_washdishes.parquet`
- Slide draft (needs reframing per above): claude.ai artifact `egoeye`
