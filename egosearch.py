"""
egosearch — quality-aware natural-language search over EgoVerse episodes.

    python egosearch.py "find successful demonstrations of placing a cup into a drawer"
    python egosearch.py "fold_clothes episodes from mecka that were fumbled" -k 10
    python egosearch.py "washing dishes under 20 seconds" --html demo_search.html

The pitch: a semantic index alone answers "what is this episode of". egoeye's
audit answers "was it done well". Neither is a training-set filter on its own.
Joined, they are: *find me clean demonstrations of X* is the query a robot-
learning team actually types, and it is the one query neither a caption index
nor a quality score can answer alone.

Three scores per hit, exactly the ones on page 2 of the brief:

  semantic  BM25 over episode task text + annotations, with domain synonym
            expansion. What the episode is of.
  signal    signal-integrity quality from eyekit's channels (eye opening,
            mask violations, rainflow retry ratio, tracking dropout).
            How clean the motion signal is. Continuous.
  success   1 - failure_score. Probability the demonstration has no discrete
            failure event (drop / collision). Event-driven.

signal and success are deliberately separate: an episode can be event-free but
smeared (hesitant, re-gripping, still bad training data), which is the case the
brief's mock result #2 illustrates — semantic 0.96 but signal 0.73.

No embedding model, no LLM, no GPU, no network. Same defensibility story as the
detector: every number here is reproducible from the parquet and this file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. Lexicon — the only hand-authored knowledge in the system.
#    Kept as data (not code) so the HTML demo can reuse it verbatim.
# ----------------------------------------------------------------------

STOPWORDS = {
    # query scaffolding — carries no retrieval signal in a 3-word task string
    "find", "show", "get", "give", "me", "us", "i", "want", "need", "looking",
    "look", "search", "please", "any", "some", "all", "a", "an", "the", "of",
    "in", "into", "on", "onto", "at", "to", "for", "from", "with", "and", "or",
    "that", "which", "where", "when", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "it", "its", "this", "these",
    "those", "by", "as", "up", "out",
    # corpus scaffolding — every episode is a demonstration, so the word is
    # pure noise: it would match everything with zero discriminative power
    "episode", "episodes", "demo", "demos", "demonstration", "demonstrations",
    "clip", "clips", "video", "videos", "recording", "recordings", "example",
    "examples", "sample", "samples", "data", "dataset", "task", "tasks",
    "human", "person", "someone", "somebody",
    # collection-protocol prefixes stamped onto thousands of task slugs
    # ('freeform_put_cup_on_saucer'): they describe how the episode was
    # recorded, never what happens in it
    "freeform", "flagship",
}

# Nouns that mean "an episode". An ambiguous quality word is read as intent
# only when one of these is nearby — see parse_query.
SCAFFOLD_NOUNS = {"episode", "episodes", "demo", "demos", "demonstration",
                  "demonstrations", "clip", "clips", "video", "videos",
                  "example", "examples", "sample", "samples", "run", "runs",
                  "take", "takes", "recording", "recordings", "data", "one",
                  "ones", "trial", "trials"}

# Domain synonyms. A match through this table scores at SYNONYM_WEIGHT of a
# literal match — near-misses should rank, but never above an exact hit.
# Groups are symmetric: every member expands to every other member.
SYNONYM_GROUPS = [
    ["place", "put", "set", "position", "insert", "load", "stow", "deposit"],
    ["cup", "mug", "glass", "tumbler", "goblet"],
    ["drawer", "cabinet", "cupboard", "shelf", "compartment"],
    ["bin", "box", "container", "crate", "tray", "basket", "carton"],
    ["bag", "sack", "pouch", "bagging"],
    ["clothes", "clothing", "laundry", "garment", "apparel", "shirt", "towel"],
    ["fold", "folding", "foldable"],
    ["wash", "washing", "clean", "cleaning", "rinse", "scrub", "wipe"],
    ["dish", "dishes", "plate", "plates", "bowl", "bowls", "saucer", "utensil",
     "cutlery", "silverware"],
    ["pack", "packing", "package", "packaging", "bagging", "boxing"],
    ["unpack", "unpacking", "unload", "unloading", "empty"],
    ["organize", "organise", "organizing", "sort", "sorting", "arrange",
     "arranging", "tidy"],
    ["prepare", "preparing", "prep", "make", "making", "cook", "cooking"],
    ["cut", "cutting", "slice", "slicing", "chop", "chopping"],
    ["pour", "pouring", "fill", "filling"],
    ["grasp", "grab", "pick", "picking", "lift", "lifting", "take"],
    ["assemble", "assembling", "build", "building", "attach", "connect"],
    ["kitchen", "countertop", "counter", "sink", "table"],
    ["iron", "ironing", "press", "pressing"],
    ["vegetable", "vegetables", "veggie", "produce", "salad"],
    ["fruit", "fruits", "apple", "orange", "banana"],
    ["grocery", "groceries", "shopping"],
    ["tool", "tools", "hardware", "screwdriver", "wrench"],
    ["fabric", "textile", "cloth", "linen"],
]

# Quality intent. These never enter the semantic match — they steer the ranking
# instead, which is the entire point of the feature.
SUCCESS_WORDS = {
    "successful", "success", "succeeded", "clean", "cleanly", "good", "great",
    "correct", "correctly", "perfect", "perfectly", "flawless", "smooth",
    "smoothly", "nominal", "proper", "properly", "best", "quality", "reliable",
    "consistent", "confident", "uninterrupted",
}
FAILURE_WORDS = {
    "failed", "failure", "failures", "fail", "fails", "failing", "bad",
    "botched", "fumble", "fumbled", "fumbles", "drop", "dropped", "drops",
    "slip", "slipped", "slips", "retry", "retries", "redo", "mistake",
    "mistakes", "error", "errors", "messy", "sloppy", "hesitant", "hesitation",
    "hesitated", "worst", "broken", "wrong", "corrected", "correction",
    "struggled", "struggle", "unstable", "jerky",
}
# "without dropping", "no failures", "didn't fail" — a failure word under
# negation is a *success* query. Two-token lookback covers every phrasing that
# actually shows up in a search box.
NEGATIONS = {"no", "not", "without", "never", "didnt", "dont", "doesnt",
             "isnt", "wasnt", "excluding", "except", "avoid", "avoiding",
             "free", "zero", "minus"}

# Words that are quality intent in one sentence and the task itself in the
# next. Measured against the corpus, not guessed: 'clean' appears inside the
# task name of 23,926 episodes (clean_kitchen, cleaning_shoes,
# clean_espresso_machine), 'smooth' in 73, 'quality' in 23. So "cleaning the
# kitchen" must not be read as a request for high-quality episodes of an
# unspecified task. These count as intent only when an episode-noun follows
# within two tokens ("clean demonstrations of ...").
AMBIGUOUS_QUALITY = {"clean", "cleans", "cleaning", "cleaned", "cleanly",
                     "smooth", "smoothly", "smoothing", "quality",
                     "correct", "correctly", "corrected", "correction"}

LABS = ["microagi", "mecka", "scale", "abc", "rl2", "eth", "song", "wang", "yam"]
# A bare "scale" is a verb as often as it is a data source, so a lab name only
# becomes a filter behind one of these — or in explicit `lab:x` form.
LAB_PREPOSITIONS = {"from", "in", "by", "at", "on", "lab", "source", "labs"}

EMBODIMENT_PATTERNS = [
    (r"\b(bimanual|two[- ]hand(ed)?|both hands)\b", "bimanual"),
    (r"\b(left[- ](hand|arm)(ed)?|one[- ]hand(ed)? left)\b", "left"),
    (r"\b(right[- ](hand|arm)(ed)?)\b", "right"),
    (r"\b(robot|eva|yam|teleop)\b", "robot"),
]

SYNONYM_WEIGHT = 0.6      # a synonym hit is worth 60% of a literal hit
BM25_K1, BM25_B = 1.2, 0.75


# ----------------------------------------------------------------------
# 2. Text normalisation
# ----------------------------------------------------------------------

_SPLIT = re.compile(r"[^a-z0-9]+")


def stem(w: str) -> str:
    """Deliberately crude suffix stripper.

    Real stemmers (Porter, Snowball) are a dependency and a behaviour we would
    then have to defend. The corpus is machine-generated task slugs — the only
    morphology that actually appears is plural-s and -ing/-ed gerunds
    ('fold' / 'folding' / 'folds'), and the synonym table covers the rest.

    Deliberately NOT stripping -er: this vocabulary is full of nouns that end
    in it and mean something else without it (water->wat, paper->pap,
    drawer->draw, container->contain, counter->count). Agent nouns are rare
    enough here that the rule loses more than it gains.
    """
    if len(w) <= 3:
        return w
    for suf, keep in (("ies", 3), ("ing", 3), ("ed", 2), ("es", 2), ("s", 1)):
        if w.endswith(suf) and len(w) - keep >= 3:
            base = w[: -keep]
            if suf == "ies":
                return base + "y"
            # doubled consonant: 'stopping' -> 'stopp' -> 'stop'
            if suf in ("ing", "ed") and len(base) > 3 and base[-1] == base[-2]:
                base = base[:-1]
            return base
    return w


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    """snake_case / free text -> stemmed content tokens."""
    toks = [t for t in _SPLIT.split(str(text).lower()) if t]
    out = []
    for t in toks:
        if drop_stop and t in STOPWORDS:
            continue
        if t.isdigit():           # 'pack_fourth_tea_box_2' — the index is noise
            continue
        out.append(stem(t))
    return out


def _build_synonym_map() -> dict[str, list[str]]:
    """stem -> other stems in its group(s)."""
    m: dict[str, set[str]] = {}
    for group in SYNONYM_GROUPS:
        stems = {stem(w) for w in group}
        for s in stems:
            m.setdefault(s, set()).update(stems - {s})
    return {k: sorted(v) for k, v in m.items()}


SYNONYMS = _build_synonym_map()

# Intent matching runs on stems as well as literals, so 'dropping' reaches
# 'drop' and 'slipped' reaches 'slip' without enumerating every inflection.
SUCCESS_STEMS = {stem(w) for w in SUCCESS_WORDS}
FAILURE_STEMS = {stem(w) for w in FAILURE_WORDS}


# ----------------------------------------------------------------------
# 3. Query understanding
# ----------------------------------------------------------------------

@dataclass
class Query:
    raw: str
    terms: list[str] = field(default_factory=list)       # stemmed content terms
    raw_terms: list[str] = field(default_factory=list)   # ...as the user typed
    intent: str = "none"                                 # success | failure | none
    intent_evidence: list[str] = field(default_factory=list)
    lab: str | None = None
    embodiment: str | None = None
    min_dur: float | None = None                          # seconds
    max_dur: float | None = None
    k: int | None = None

    def describe(self) -> str:
        bits = [f"terms={self.raw_terms or '[]'}"]
        if self.intent != "none":
            bits.append(f"want={self.intent} ({'+'.join(self.intent_evidence)})")
        if self.lab:
            bits.append(f"lab={self.lab}")
        if self.embodiment:
            bits.append(f"embodiment={self.embodiment}")
        if self.min_dur is not None:
            bits.append(f"duration>={self.min_dur:g}s")
        if self.max_dur is not None:
            bits.append(f"duration<={self.max_dur:g}s")
        return "  ".join(bits)


_DUR_UNIT = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
             "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60}
_UNDER = r"(?:under|below|less than|shorter than|at most|<=?|max)"
_OVER = r"(?:over|above|more than|longer than|at least|>=?|min)"
_NUMUNIT = r"(\d+(?:\.\d+)?)\s*(s|secs?|seconds?|m|mins?|minutes?)\b"


def parse_query(raw: str) -> Query:
    """Natural language -> intent + filters + content terms.

    Rule-based on purpose. An LLM parser here would reintroduce exactly the
    nondeterminism the whole project is arguing against, and the query language
    of a robotics dataset is small enough to enumerate.
    """
    q = Query(raw=raw)
    text = " " + raw.lower().replace("’", "'").replace("'", "") + " "

    # --- explicit field:value operators (power-user path) ---
    for m in re.finditer(r"\b(lab|source)\s*:\s*([a-z0-9_]+)", text):
        q.lab = m.group(2)
    text = re.sub(r"\b(lab|source)\s*:\s*[a-z0-9_]+", " ", text)

    # --- duration filters (before tokenising: they are multi-word) ---
    m = re.search(_UNDER + r"\s*" + _NUMUNIT, text)
    if m:
        q.max_dur = float(m.group(1)) * _DUR_UNIT[m.group(2)]
    m = re.search(_OVER + r"\s*" + _NUMUNIT, text)
    if m:
        q.min_dur = float(m.group(1)) * _DUR_UNIT[m.group(2)]
    if q.max_dur is None and q.min_dur is None:
        if re.search(r"\b(short|brief|quick)\b", text):
            q.max_dur = 15.0            # ~corpus median is 13.6s
        elif re.search(r"\b(long|lengthy|extended)\b", text):
            q.min_dur = 60.0
    text = re.sub(_UNDER + r"\s*" + _NUMUNIT, " ", text)
    text = re.sub(_OVER + r"\s*" + _NUMUNIT, " ", text)
    text = re.sub(r"\b(short|brief|quick|long|lengthy|extended)\b", " ", text)

    # --- result count: "top 5", "best 20 episodes" ---
    m = re.search(r"\b(?:top|first|best|worst)\s+(\d{1,3})\b", text)
    if m:
        q.k = int(m.group(1))

    # --- embodiment ---
    for pat, name in EMBODIMENT_PATTERNS:
        if re.search(pat, text):
            q.embodiment = name
            text = re.sub(pat, " ", text)
            break

    words = [w for w in _SPLIT.split(text) if w]

    # --- lab, only behind a preposition (a bare 'scale' stays a content word) ---
    if q.lab is None:
        for i, w in enumerate(words):
            if w in LABS and i > 0 and words[i - 1] in LAB_PREPOSITIONS:
                q.lab = w
                words[i] = ""
                break

    # --- quality intent, with negation flip ---
    pos, neg = [], []
    consumed_idx: set[int] = set()
    for i, w in enumerate(words):
        if not w:
            continue
        s = stem(w)
        is_success = w in SUCCESS_WORDS or s in SUCCESS_STEMS
        is_failure = w in FAILURE_WORDS or s in FAILURE_STEMS
        if w in AMBIGUOUS_QUALITY and not (
                set(words[i + 1: i + 3]) & SCAFFOLD_NOUNS):
            is_success = is_failure = False     # it is the task, not the intent
        if not (is_success or is_failure):
            continue
        consumed_idx.add(i)
        # "without dropping", "no failures", "didn't fail" — a failure word
        # under negation is a success query
        negated = any(p in NEGATIONS for p in words[max(0, i - 2): i])
        if is_success:
            (neg if negated else pos).append(w)
        else:
            (pos if negated else neg).append(w)
    if pos and not neg:
        q.intent, q.intent_evidence = "success", pos
    elif neg and not pos:
        q.intent, q.intent_evidence = "failure", neg
    elif pos and neg:
        # mixed signals ("successful, no drops") — the majority wins, and a tie
        # means the user asked for both, which is the same as asking for neither
        q.intent = "success" if len(pos) >= len(neg) else "failure"
        q.intent_evidence = pos + neg

    # --- content terms: everything the filters did not consume ---
    # consumed_idx is positional, not a word set: an ambiguous word that was
    # ruled a task verb ("cleaning the kitchen") has to survive as a term, even
    # when the same query also used it as intent ("clean demos of cleaning...")
    drop = NEGATIONS | {"top", "first", "best", "worst"}
    q.raw_terms = [w for i, w in enumerate(words)
                   if w and i not in consumed_idx and w not in STOPWORDS
                   and w not in drop and not w.isdigit()]
    q.terms = [stem(w) for w in q.raw_terms]
    return q


# ----------------------------------------------------------------------
# 4. Quality channels — the "signal" half of quality-aware
# ----------------------------------------------------------------------

# channel -> (weight, mapping from the raw eyekit column to "how clean" in [0,1])
SIGNAL_CHANNELS = {
    "eye_opening":        (0.40, lambda v: np.clip(v, 0, 1)),
    "mask_violation_p90": (0.20, lambda v: 1 - np.clip(v / 0.30, 0, 1)),
    # same mapping as eyekit's retry channel, inverted: dithering is dirt
    "rf_small_ratio":     (0.25, lambda v: 1 - np.clip((v - 0.05) / 0.30, 0, 1)),
    "nan_frac":           (0.15, lambda v: 1 - np.clip(v / 0.30, 0, 1)),
}


def signal_scores(df: pd.DataFrame) -> np.ndarray:
    """Signal-integrity quality in [0,1] per row; 1 = a textbook-clean trace.

    Four channels, each mapped to "how clean", each degrading gracefully to
    absent (short episodes never get an eye diagram; sparse traces never get a
    rainflow decomposition). Weights renormalise over whatever survived, which
    is the same policy score_episode() uses — so an episode is never penalised
    for a channel that could not be computed. All-absent -> NaN, never 0:
    "we did not measure this" and "we measured this and it was terrible" must
    not be the same number.
    """
    n = len(df)
    num = np.zeros(n, dtype=float)
    den = np.zeros(n, dtype=float)
    for col, (w, fn) in SIGNAL_CHANNELS.items():
        if col not in df:
            continue
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(v)
        if not ok.any():
            continue
        num[ok] += w * np.asarray(fn(v[ok]), dtype=float)
        den[ok] += w
    out = np.full(n, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def calibrate(raw: np.ndarray) -> np.ndarray:
    """Percentile-rank the raw signal composite within the audited corpus.

    Why this is not gilding: on the first 50 real episodes rf_small_ratio runs
    0.61–0.95 (the synthetic mock assumed ~0.10) and mask_violation_p90 has a
    median of 0.46. Both absolute mappings therefore saturate, and the raw
    composite collapses into the band 0.15–0.42 — every real episode looks
    equally mediocre and the channel stops discriminating, which is precisely
    when a search ranking needs it most.

    Percentile rank is monotone, so it never reorders episodes on signal alone;
    what it fixes is the blend against `success`, where a saturated signal term
    would otherwise contribute a near-constant and hand the whole ranking to
    the impulse channel. Reading: 0.97 = cleaner than 97% of what we audited.
    Absolute thresholds return the moment the channel mappings are recalibrated
    against real data (--signal-scale absolute).
    """
    out = np.full(len(raw), np.nan)
    ok = np.isfinite(raw)
    n = int(ok.sum())
    if n == 0:
        return out
    if n < 30:
        # too few points for a stable percentile; absolute is the safer read
        return np.clip(raw, 0, 1)
    ranks = pd.Series(raw[ok]).rank(method="average").to_numpy()
    out[ok] = (ranks - 0.5) / n
    return out


def success_scores(df: pd.DataFrame) -> np.ndarray:
    """1 - failure_score. The discrete-event channel, straight from the audit."""
    if "failure_score" not in df:
        return np.full(len(df), np.nan)
    fs = pd.to_numeric(df["failure_score"], errors="coerce").to_numpy(dtype=float)
    return 1.0 - np.clip(fs, 0, 1)


def event_rate_scores(df: pd.DataFrame, ref: float | None = None
                      ) -> tuple[np.ndarray, float]:
    """Length-normalised success: 1 - (impulses per minute / reference rate).

    THE BUG THIS FIXES. failure_score is 60% weighted on raw impulse *count*,
    and impulses fire at a roughly constant background rate — so the count is
    mostly a measure of how long the episode is. On the audited episodes here
    corr(duration, n_impulses) = 0.66, and it showed up directly in the
    rankings: "clean examples of washing dishes" returned a mean duration of
    9.0s while "wash dishes that were fumbled" returned 34.6s. The search was
    sorting by length and calling it quality. (Person C's prevalence work hit
    the same artifact from the other side, which is why the headline number is
    reported per minute rather than per episode.)

    Dividing by duration removes it: a 90s episode with one impulse (0.7/min)
    is now cleaner than a 6s episode with one impulse (10/min), which is the
    correct direction and the opposite of what the raw count says.

    The reference rate is the 90th percentile of the non-zero rates in this
    corpus rather than a constant, so it recalibrates itself against whatever
    the detector's thresholds are doing that hour instead of hard-coding a
    number that a re-freeze would silently invalidate.

    This does NOT redefine the detector. failure_score and failure_flag are
    untouched and still reported; this is the ranking's own quality channel,
    and it inherits whatever Person B lands.
    """
    n = len(df)
    if "n_impulses" not in df:
        return np.full(n, np.nan), float("nan")
    imp = pd.to_numeric(df["n_impulses"], errors="coerce").to_numpy(dtype=float)
    minutes = pd.to_numeric(df.get("duration_s"), errors="coerce").to_numpy(
        dtype=float) / 60.0
    rate = np.full(n, np.nan)
    ok = np.isfinite(imp) & np.isfinite(minutes) & (minutes > 0)
    rate[ok] = imp[ok] / minutes[ok]

    if ref is None or not np.isfinite(ref) or ref <= 0:
        nz = rate[np.isfinite(rate) & (rate > 0)]
        if len(nz) >= 10:
            ref = float(np.percentile(nz, 90))
        elif len(nz) >= 3:
            # too few points for a percentile — span the observed range instead,
            # which still orders episodes by density even on a 50-episode
            # smoke-test parquet. Provisional: it moves as the audit grows.
            ref = float(nz.max())
        else:
            ref = float("nan")
    if not np.isfinite(ref) or ref <= 0:
        # nothing to normalise against: fall back to the detector's own rule,
        # length bias and all, rather than inventing a reference
        return np.full(n, np.nan), float("nan")

    out = np.full(n, np.nan)
    good = np.isfinite(rate)
    out[good] = 1.0 - np.clip(rate[good] / ref, 0, 1)
    return out, ref


def signal_score(row) -> float:
    """Single-row convenience wrapper (tests, notebooks)."""
    return float(signal_scores(pd.DataFrame([dict(row)]))[0])


def success_score(row) -> float:
    return float(success_scores(pd.DataFrame([dict(row)]))[0])


def _num(v) -> float:
    try:
        if v is None:
            return float("nan")
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ----------------------------------------------------------------------
# 5. The index
# ----------------------------------------------------------------------

@dataclass
class Hit:
    rank: int
    episode_id: str
    task: str
    lab: str
    embodiment: str
    duration_s: float
    semantic: float
    signal: float
    signal_raw: float
    success: float
    success_raw: float
    score: float
    scored: bool                      # False = audit has not reached it yet
    n_impulses: float = float("nan")
    impulse_times: list = field(default_factory=list)
    eye_opening: float = float("nan")
    rf_small_ratio: float = float("nan")
    zarr_path: str = ""
    text: str = ""                    # annotation text, when available

    def to_dict(self):
        return asdict(self)


class EgoSearch:
    """BM25 over unique task strings, joined to per-episode audit scores.

    Why index unique task strings rather than episodes: 438,874 episodes share
    27,996 distinct task texts. Scoring the distinct texts is 16x less work and
    gives identical results, because every episode of the same task is
    semantically identical by construction — which is exactly why the *quality*
    channel is what orders them. That is the thesis of the feature in one line.
    """

    def __init__(self, df: pd.DataFrame, signal_scale: str = "percentile",
                 success_scale: str = "rate"):
        self.df = df.reset_index(drop=True)
        self.signal_scale = signal_scale
        self.success_scale = success_scale

        # Quality is a property of the episode, not of the query — compute it
        # once here rather than per search over the matching subset.
        self.df["_signal_raw"] = signal_scores(self.df)
        self.df["_signal"] = (calibrate(self.df["_signal_raw"].to_numpy())
                              if signal_scale == "percentile"
                              else np.clip(self.df["_signal_raw"].to_numpy(), 0, 1))
        self.df["_success_raw"] = success_scores(self.df)
        rate, self.rate_ref = event_rate_scores(self.df)
        self.df["_success_rate"] = rate
        use_rate = success_scale == "rate" and np.isfinite(self.rate_ref)
        self.success_scale = "rate" if use_rate else "raw"
        self.df["_success"] = rate if use_rate else self.df["_success_raw"]

        docs = self.df["_doc"].astype(str)
        uniq, inverse = np.unique(docs.to_numpy(), return_inverse=True)
        self.doc_text = list(uniq)
        self.df["_doc_id"] = inverse

        # postings: token -> {doc_id: tf}
        self.postings: dict[str, dict[int, int]] = {}
        self.doc_len = np.zeros(len(uniq), dtype=np.float32)
        for i, text in enumerate(self.doc_text):
            toks = tokenize(text)
            self.doc_len[i] = len(toks) or 1
            for t in toks:
                self.postings.setdefault(t, {}).setdefault(i, 0)
                self.postings[t][i] += 1
        self.avg_dl = float(self.doc_len.mean()) if len(self.doc_len) else 1.0

        n = len(self.doc_text)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

        # doc_id -> episode row indices, so a doc score fans out in one hop
        order = np.argsort(self.df["_doc_id"].to_numpy(), kind="stable")
        self._order = order
        self._starts = np.searchsorted(
            self.df["_doc_id"].to_numpy()[order], np.arange(n + 1))

    # -- construction -------------------------------------------------
    @classmethod
    def build(cls, episodes_csv: str, results_parquet: str | None = None,
              annotations_csv: str | None = None, scope: str = "auto",
              verbose: bool = True,
              signal_scale: str = "percentile",
              success_scale: str = "rate") -> "EgoSearch":
        ep = pd.read_csv(episodes_csv, dtype={"episode_id": str})
        ep["task"] = ep["task"].fillna("").astype(str)
        ep["lab"] = ep["lab"].fillna("").astype(str)
        ep["embodiment"] = ep["embodiment"].fillna("").astype(str)
        ep["duration_s"] = (pd.to_numeric(ep["n_frames"], errors="coerce")
                            / pd.to_numeric(ep["fps"], errors="coerce"))

        res = None
        if results_parquet:
            res = pd.read_parquet(results_parquet)
            res["episode_id"] = res["episode_id"].astype(str)
            if "error" in res:
                # error rows carry no scores; they must not be ranked as if
                # they were clean, and they must not silently vanish either
                res = res[res["error"].fillna("") == ""]
            keep = [c for c in ["episode_id", "failure_score", "failure_flag",
                                "n_impulses", "impulse_frames", "eye_opening",
                                "rf_small_ratio", "mask_violation_p90",
                                "nan_frac"] if c in res.columns]
            res = res[keep].drop_duplicates("episode_id")
            ep = ep.merge(res, on="episode_id", how="left")
            ep["_scored"] = ep["failure_score"].notna()
        else:
            ep["_scored"] = False

        if annotations_csv:
            try:
                ann = pd.read_csv(annotations_csv, dtype={"episode_id": str})
                if "text" in ann:
                    ann["text"] = ann["text"].fillna("").astype(str)
                    ep = ep.merge(ann[["episode_id", "text"]], on="episode_id",
                                  how="left")
            except FileNotFoundError:
                if verbose:
                    print(f"note: {annotations_csv} not found — searching task "
                          "text only", file=sys.stderr)
        if "text" not in ep:
            ep["text"] = ""
        ep["text"] = ep["text"].fillna("").astype(str)

        if scope == "scored":
            ep = ep[ep["_scored"]]

        # The searchable document. Annotation text is appended when present:
        # free-text captions are far richer than a task slug, and they are the
        # only place a phrase like "into a drawer" is ever written out.
        ep["_doc"] = (ep["task"].str.replace("_", " ", regex=False)
                      + " " + ep["text"]).str.strip()

        if verbose:
            n_scored = int(ep["_scored"].sum())
            print(f"index: {len(ep):,} episodes | {ep['_doc'].nunique():,} "
                  f"distinct texts | {n_scored:,} with audit scores "
                  f"({n_scored / max(len(ep), 1):.1%})", file=sys.stderr)
            _warn_saturated(ep)
            _warn_length_bias(ep)
        return cls(ep, signal_scale=signal_scale, success_scale=success_scale)

    # -- retrieval ----------------------------------------------------
    def _score_docs(self, terms: list[str]) -> np.ndarray:
        """BM25 with synonym expansion, normalised to [0,1].

        Normalised against the *ideal* document (one that contains every query
        term exactly once at average length) rather than against the best hit
        in this result set — so 0.94 means the same thing across two different
        queries, which it would not if we min-maxed per query.
        """
        scores = np.zeros(len(self.doc_text), dtype=np.float32)
        if not terms:
            return scores
        ideal = 0.0
        for term in terms:
            # a query term contributes through itself at full weight and
            # through each synonym at SYNONYM_WEIGHT; a doc containing both
            # takes the max, not the sum, so synonyms cannot double-count
            variants = [(term, 1.0)] + [(s, SYNONYM_WEIGHT)
                                        for s in SYNONYMS.get(term, [])]
            # normalise against the LITERAL term, so a synonym hit lands below
            # 1.0 by construction. Falling back to the best synonym only when
            # the literal term is absent from the corpus keeps an
            # out-of-vocabulary word ('mug' when the corpus only says 'cup')
            # from collapsing the whole denominator to zero.
            base_idf = self.idf.get(term)
            if base_idf is None:
                base_idf = max((self.idf.get(v, 0.0) * w for v, w in variants),
                               default=0.0)
            # tf=1 at average document length: idf * (k1+1)/(1+k1) == idf
            ideal += base_idf
            term_scores = np.zeros_like(scores)
            for variant, weight in variants:
                posting = self.postings.get(variant)
                if not posting:
                    continue
                idf = self.idf[variant] * weight
                ids = np.fromiter(posting.keys(), dtype=np.int64, count=len(posting))
                tfs = np.fromiter(posting.values(), dtype=np.float32, count=len(posting))
                dl = self.doc_len[ids]
                denom = tfs + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avg_dl)
                contrib = idf * tfs * (BM25_K1 + 1) / denom
                np.maximum.at(term_scores, ids, contrib)
            scores += term_scores
        if ideal <= 0:
            return scores
        scores /= ideal

        # Coverage gate: a doc matching 1 of 3 query terms is a worse answer
        # than BM25's additive score implies, because IDF rewards the rare term
        # regardless of whether the rest of the query was honoured at all.
        # ("cup" alone must not outrank "cup" + "drawer".)
        if len(terms) > 1:
            cover = np.zeros(len(self.doc_text), dtype=np.float32)
            for term in terms:
                hit = np.zeros(len(self.doc_text), dtype=bool)
                for variant, _w in [(term, 1.0)] + [(s, SYNONYM_WEIGHT)
                                                    for s in SYNONYMS.get(term, [])]:
                    posting = self.postings.get(variant)
                    if posting:
                        hit[np.fromiter(posting.keys(), dtype=np.int64,
                                        count=len(posting))] = True
                cover += hit
            cover /= len(terms)
            scores *= (0.35 + 0.65 * cover)
        return np.clip(scores, 0, 1)

    def search(self, raw_query: str, k: int = 10, quality_weight: float = 0.75,
               scope: str = "auto") -> tuple[Query, list[Hit]]:
        q = parse_query(raw_query)
        k = q.k or k

        df = self.df
        mask = np.ones(len(df), dtype=bool)
        if q.lab:
            mask &= (df["lab"].str.lower() == q.lab).to_numpy()
        if q.embodiment:
            emb = df["embodiment"].str.lower()
            if q.embodiment == "robot":
                mask &= ~emb.str.startswith("human").to_numpy()
            elif q.embodiment == "bimanual":
                mask &= emb.str.contains("bimanual").to_numpy()
            else:
                mask &= emb.str.contains(q.embodiment).to_numpy()
        dur = df["duration_s"].to_numpy(dtype=float)
        if q.min_dur is not None:
            mask &= np.nan_to_num(dur, nan=-1) >= q.min_dur
        if q.max_dur is not None:
            mask &= (np.nan_to_num(dur, nan=1e9) <= q.max_dur)

        doc_scores = self._score_docs(q.terms)
        sem = doc_scores[df["_doc_id"].to_numpy()]
        if q.terms:
            mask &= sem > 0

        # One merged ranking, audited and unaudited together. The tempting
        # alternative — audited episodes first, unaudited only as backfill —
        # buries a perfect text match behind a barely-relevant one just because
        # the fan-out happened to reach it, and today the fan-out has reached
        # <1% of the catalogue. Unaudited episodes instead take quality=0.5 in
        # the fusion (see _fuse): explicitly agnostic, so they land *between*
        # a measured-clean and a measured-fumbled episode of the same task and
        # are labelled UNAUDITED in every output.
        if scope == "scored":
            mask &= df["_scored"].to_numpy(dtype=bool)

        idx = np.flatnonzero(mask)
        if not len(idx):
            return q, []
        sub = df.iloc[idx]
        sig = sub["_signal"].to_numpy(dtype=float)
        suc = sub["_success"].to_numpy(dtype=float)
        final = self._fuse(sem[idx], sig, suc, q.intent, quality_weight)

        # Ties are common and they are not noise: every episode of a task
        # shares a semantic score, and unaudited ones share a quality score
        # too. Break them on episode_id so the ranking is a function of the
        # data alone — not of row order, which differs between this index and
        # the one the demo page rebuilds in the browser.
        ids = sub["episode_id"].to_numpy(dtype=str)
        order = np.lexsort((ids, -final))[:k]
        recs = sub.iloc[order].to_dict("records")
        hits: list[Hit] = []
        for rank, (j, row) in enumerate(zip(order, recs), start=1):
            i = idx[j]
            hits.append(Hit(
                rank=rank,
                episode_id=str(row["episode_id"]),
                task=str(row["task"]),
                lab=str(row["lab"]),
                embodiment=str(row["embodiment"]),
                duration_s=_num(row["duration_s"]),
                semantic=float(sem[i]),
                signal=float(sig[j]),
                signal_raw=_num(row.get("_signal_raw")),
                success=float(suc[j]),
                success_raw=_num(row.get("_success_raw")),
                score=float(final[j]),
                scored=bool(row["_scored"]),
                n_impulses=_num(row.get("n_impulses")),
                impulse_times=_impulse_times(row),
                eye_opening=_num(row.get("eye_opening")),
                rf_small_ratio=_num(row.get("rf_small_ratio")),
                zarr_path=str(row.get("zarr_path", "")),
                text=str(row.get("text", "")),
            ))
        return q, hits

    @staticmethod
    def _fuse(sem: np.ndarray, sig: np.ndarray, suc: np.ndarray, intent: str,
              quality_weight: float) -> np.ndarray:
        """final = semantic * quality^w.

        Multiplicative, not additive: a query is *about something*, so an
        irrelevant-but-pristine episode must never surface. Quality re-orders
        within relevance, it does not substitute for it. w=0 reduces this to
        plain semantic search, which is what a query with no stated quality
        intent should get.
        """
        sem = np.asarray(sem, dtype=float)
        if intent == "none" or quality_weight <= 0:
            return sem
        # unmeasured channels fall back to 0.5 — explicitly agnostic, so an
        # unaudited episode is neither rewarded nor punished for the gap
        sig = np.where(np.isfinite(sig), sig, 0.5)
        suc = np.where(np.isfinite(suc), suc, 0.5)
        if intent == "success":
            q = 0.65 * suc + 0.35 * sig
        else:                                  # find me the bad ones
            q = 0.65 * (1 - suc) + 0.35 * (1 - sig)
        return sem * np.clip(q, 1e-6, 1.0) ** quality_weight


def _warn_saturated(ep: pd.DataFrame) -> None:
    """Shout if a quality channel has pinned — a channel that reads the same on
    every episode ranks nothing, and it is invisible in the parquet.

    This is a real condition on the first 50 real episodes, not a hypothetical:
    rf_small_ratio lands past the top of eyekit's [0.05, 0.35] mapping on all
    of them, which also puts a constant 0.25 floor under every failure_score.
    """
    scored = ep[ep["_scored"]]
    if len(scored) < 20:
        return
    for col, hi in (("rf_small_ratio", 0.35), ("mask_violation_p90", 0.30)):
        if col not in scored:
            continue
        v = pd.to_numeric(scored[col], errors="coerce").dropna()
        if len(v) >= 20 and (v >= hi).mean() > 0.9:
            print(f"WARNING: {col} is past its mapping ceiling ({hi}) on "
                  f"{(v >= hi).mean():.0%} of audited episodes (median "
                  f"{v.median():.2f}) — that channel is saturated and "
                  f"discriminates nothing. Signal falls back to percentile "
                  f"ranking; tell whoever owns eyekit's thresholds.",
                  file=sys.stderr)


def _warn_length_bias(ep: pd.DataFrame) -> None:
    """Report how much of the impulse count is just episode duration."""
    scored = ep[ep["_scored"]]
    if len(scored) < 20 or "n_impulses" not in scored:
        return
    d = pd.to_numeric(scored["duration_s"], errors="coerce")
    i = pd.to_numeric(scored["n_impulses"], errors="coerce")
    ok = d.notna() & i.notna() & (d > 0)
    if ok.sum() < 20 or i[ok].nunique() < 2:
        return
    r = float(d[ok].corr(i[ok]))
    if np.isfinite(r) and r > 0.4:
        print(f"NOTE: corr(duration, n_impulses) = {r:.2f} on audited episodes "
              f"— impulse count is partly a measure of episode length. Ranking "
              f"uses impulses per minute instead (--success-scale raw to "
              f"disable).", file=sys.stderr)


def _impulse_times(row) -> list:
    """Impulse frame indices -> seconds. These are the frames where the
    confidence meter dips; the demo page draws a tick at each one."""
    frames = row.get("impulse_frames")
    if frames is None or isinstance(frames, float):
        return []
    fps = _num(row.get("fps")) or 30.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    try:
        return [round(float(f) / fps, 2) for f in list(frames)][:40]
    except TypeError:
        return []


# ----------------------------------------------------------------------
# 6. CLI
# ----------------------------------------------------------------------

def _bar(v: float, width: int = 18) -> str:
    if not np.isfinite(v):
        return "·" * width
    n = int(round(np.clip(v, 0, 1) * width))
    return "█" * n + "░" * (width - n)


def _fmt(v: float) -> str:
    return "  — " if not np.isfinite(v) else f"{v:.2f}"


def print_hits(q: Query, hits: list[Hit], quality_weight: float) -> None:
    print(f'\nquery   "{q.raw}"')
    print(f"parsed  {q.describe()}")
    if q.intent == "none":
        print("        (no quality intent stated — ranking on relevance alone; "
              'try "successful ..." or "fumbled ...")')
    print(f"        quality weight w={quality_weight:g}\n")
    if not hits:
        print("no matches. Widen the terms, or the audit has not reached this "
              "task family yet (--scope all).")
        return
    for h in hits:
        dur = "—" if not np.isfinite(h.duration_s) else f"{h.duration_s:5.1f}s"
        tag = "" if h.scored else "   [UNAUDITED — no quality evidence]"
        print(f"#{h.rank:<2} {h.task[:52]:<52} {dur}  {h.lab}{tag}")
        print(f"    {h.episode_id}   final {h.score:.3f}")
        print(f"    semantic {_bar(h.semantic)} {_fmt(h.semantic)}"
              f"    signal {_bar(h.signal)} {_fmt(h.signal)}"
              f"    success {_bar(h.success)} {_fmt(h.success)}")
        if h.impulse_times:
            print(f"    impulse events at {h.impulse_times[:8]}s "
                  f"({int(h.n_impulses)} total)")
        if h.text:
            print(f"    “{h.text[:110]}”")
        print()


# Order matters: the first one is what the demo page loads with, so it has to
# be a query the audit has actually covered. The brief's flagship cup/drawer
# example comes second — it demonstrates retrieval across the full 438k
# catalogue, but every hit is currently unaudited, which is the least
# convincing thing to open on.
DEMO_QUERIES = [
    "clean examples of washing dishes",
    "wash dishes episodes that were fumbled",
    "find successful demonstrations of placing a cup into a drawer",
    "successful folding clothes from mecka",
    "packing groceries into bags without dropping anything",
    "short bimanual kitchen demonstrations, no mistakes",
]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Quality-aware natural-language search over EgoVerse episodes.")
    ap.add_argument("query", nargs="*", help="natural language query")
    ap.add_argument("--episodes", default="episodes.csv")
    ap.add_argument("--results", default="audit_results.parquet",
                    help="audit parquet; quality ranking is inert without it")
    ap.add_argument("--annotations", default="annotations.csv",
                    help="optional episode_id,text[,preview_mp4] export")
    ap.add_argument("-k", "--top", type=int, default=10)
    ap.add_argument("-w", "--quality-weight", type=float, default=0.75,
                    help="0 = pure semantic search, 1.5 = quality-dominated")
    ap.add_argument("--scope", choices=["auto", "scored", "all"], default="auto")
    ap.add_argument("--signal-scale", choices=["percentile", "absolute"],
                    default="percentile",
                    help="percentile = rank within the audited corpus (robust "
                         "to saturated channels); absolute = fixed thresholds")
    ap.add_argument("--success-scale", choices=["rate", "raw"], default="rate",
                    help="rate = impulses per minute (removes the length bias "
                         "in the raw impulse count); raw = 1 - failure_score")
    ap.add_argument("--json", metavar="PATH", help="write hits as JSON")
    ap.add_argument("--html", metavar="PATH", help="write the demo page")
    ap.add_argument("--demo", action="store_true",
                    help="run the canned query set (use with --html)")
    args = ap.parse_args(argv)

    import os
    results = args.results if args.results and os.path.exists(args.results) else None
    if args.results and results is None:
        print(f"note: {args.results} not found — quality channels will be empty. "
              "Run the Modal audit first.", file=sys.stderr)
    annotations = (args.annotations
                   if args.annotations and os.path.exists(args.annotations) else None)

    index = EgoSearch.build(args.episodes, results, annotations, scope="all",
                            signal_scale=args.signal_scale,
                            success_scale=args.success_scale)

    queries = [" ".join(args.query)] if args.query else []
    if args.demo or not queries:
        queries = DEMO_QUERIES if not queries else queries + DEMO_QUERIES

    runs = []
    for raw in queries:
        q, hits = index.search(raw, k=args.top,
                               quality_weight=args.quality_weight,
                               scope=args.scope)
        print_hits(q, hits, args.quality_weight)
        runs.append((q, hits))

    if args.json:
        payload = [{"query": q.raw, "parsed": q.describe(),
                    "hits": [h.to_dict() for h in hits]} for q, hits in runs]
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"wrote {args.json}")

    if args.html:
        from search_page import write_page
        write_page(args.html, index, runs, args.quality_weight,
                   source=results or "no audit parquet")
        print(f"wrote {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
