"""Renders the quality-aware search demo as a single self-contained HTML file.

    python egosearch.py --demo --results audit_results.parquet --html demo_search.html

The page runs the same engine client-side so a judge can type their own query.
The lexicon (stopwords, synonyms, quality words, negations) is exported from
egosearch.py rather than restated in JavaScript, so there is exactly one place
to change a synonym. The ~120 lines of scoring JS are a transliteration of
EgoSearch._score_docs / _fuse — which is a real divergence risk, so the page
re-runs every canned query in the browser and compares its top-5 against the
Python result embedded in the payload. That comparison is the "engine parity"
badge in the footer; if it ever reads anything but all-match, trust the CLI.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import egosearch as E

# One representative episode per distinct task guarantees that any query a
# judge invents returns something, at ~45 bytes each. Every audited episode is
# included on top of that, because those are the ones with quality evidence.
MAX_LITE = 30000


def _payload(index: E.EgoSearch, runs, quality_weight: float) -> dict:
    df = index.df
    scored_mask = df["_scored"].to_numpy(dtype=bool)

    # --- which episodes travel with the page ---
    keep = set(np.flatnonzero(scored_mask).tolist())
    # one per distinct document, preferring an audited one where it exists
    first_of_doc = (df.assign(_i=np.arange(len(df)))
                      .sort_values(["_doc_id", "_scored"], ascending=[True, False])
                      .drop_duplicates("_doc_id")["_i"].to_numpy())
    keep.update(first_of_doc[:MAX_LITE].tolist())
    # plus everything the canned queries actually returned, so the demo is exact
    ids_wanted = {h.episode_id for _q, hits in runs for h in hits}
    if ids_wanted:
        keep.update(np.flatnonzero(df["episode_id"].isin(ids_wanted).to_numpy()).tolist())

    sel = df.iloc[sorted(keep)]

    docs = [str(t) for t in index.doc_text]
    labs = sorted(set(sel["lab"].astype(str)))
    embs = sorted(set(sel["embodiment"].astype(str)))
    lab_ix = {v: i for i, v in enumerate(labs)}
    emb_ix = {v: i for i, v in enumerate(embs)}

    def dur(v):
        v = E._num(v)
        return round(float(v), 1) if np.isfinite(v) else None

    # 6 decimals. The quality channels are what order episodes that share a
    # task, and adjacent episodes there routinely differ by ~1e-5 in the final
    # score — round to 3 or 4 and the page manufactures ties the CLI does not
    # have, which the engine-parity badge then reports as a mismatch. The cost
    # is ~10% payload; the benefit is that the badge means something.
    def r2(v, nd=6):
        v = E._num(v)
        return round(float(v), nd) if np.isfinite(v) else None

    lite, full = [], []
    for row in sel.to_dict("records"):
        base = [str(row["episode_id"]), int(row["_doc_id"]),
                lab_ix[str(row["lab"])], emb_ix[str(row["embodiment"])],
                dur(row["duration_s"]), str(row["task"])]
        if row["_scored"]:
            # signal travels precomputed rather than recomputed in JS: it is a
            # percentile within the audited corpus, so the browser could not
            # reproduce it from the shipped subset anyway, and shipping it
            # deletes a whole class of engine-parity drift
            full.append(base + [
                r2(row.get("_signal")), r2(row.get("_signal_raw")),
                r2(row.get("_success")), r2(row.get("_success_raw")),
                r2(row.get("failure_score")), r2(row.get("eye_opening")),
                r2(row.get("rf_small_ratio")), r2(row.get("mask_violation_p90")),
                r2(row.get("nan_frac")), r2(row.get("n_impulses"), 0),
                E._impulse_times(row)[:24],
            ])
        else:
            lite.append(base)

    reference = [{"q": q.raw, "top": [h.episode_id for h in hits[:5]]}
                 for q, hits in runs]

    n_scored = int(scored_mask.sum())
    aud = df[scored_mask]
    dur_corr = float("nan")
    if len(aud) >= 20 and "n_impulses" in aud:
        d = pd.to_numeric(aud["duration_s"], errors="coerce")
        i = pd.to_numeric(aud["n_impulses"], errors="coerce")
        if i.nunique() > 1:
            dur_corr = float(d.corr(i))
    return {
        "successScale": index.success_scale,
        "rateRef": (round(index.rate_ref, 2)
                    if np.isfinite(index.rate_ref) else None),
        "durCorr": (round(dur_corr, 2) if np.isfinite(dur_corr) else None),
        "docs": docs, "labs": labs, "embs": embs,
        "lite": lite, "full": full,
        "reference": reference,
        "qw": quality_weight,
        "stats": {
            "episodes_total": int(len(df)),
            "docs_total": int(len(docs)),
            "audited": n_scored,
            "shipped": int(len(sel)),
        },
        "lex": {
            "stop": sorted(E.STOPWORDS),
            "scaffold": sorted(E.SCAFFOLD_NOUNS),
            "syn": E.SYNONYMS,
            "success": sorted(E.SUCCESS_WORDS),
            "failure": sorted(E.FAILURE_WORDS),
            "successStems": sorted(E.SUCCESS_STEMS),
            "failureStems": sorted(E.FAILURE_STEMS),
            "neg": sorted(E.NEGATIONS),
            "ambiguous": sorted(E.AMBIGUOUS_QUALITY),
            "labs": E.LABS,
            "labPreps": sorted(E.LAB_PREPOSITIONS),
            "synWeight": E.SYNONYM_WEIGHT,
            "k1": E.BM25_K1, "b": E.BM25_B,
        },
    }


def write_page(path: str, index: E.EgoSearch, runs, quality_weight: float,
               source: str = "", synthetic: bool | None = None) -> None:
    data = _payload(index, runs, quality_weight)
    data["source"] = source
    # make_mock_parquet.py prints "these numbers are INVENTED. Never put them
    # on a slide" and then the file gets passed around like any other parquet.
    # Filename is the only marker it leaves, so key off it and let the caller
    # override when a real run is named something unlucky.
    data["synthetic"] = synthetic if synthetic is not None else (
        "mock" in source.lower() or "fake" in source.lower()
        or "synthetic" in source.lower() or not source
        or source == "no audit parquet")
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(data, separators=(",", ":")))
    with open(path, "w") as f:
        f.write(html)


TEMPLATE = r"""<title>egoeye Search Console</title>
<style>
  /* ---- tokens: instrument bench. Neutrals carry a faint cyan bias so they
     sit under the trace colour rather than fighting it. ---- */
  :root{
    --ground:#E9EDEC; --panel:#FDFEFE; --panel-2:#F1F5F4; --sunk:#DFE6E5;
    --ink:#101619; --ink-2:#48585C; --ink-3:#75868A;
    --rule:#D2DCDA; --rule-2:#BAC8C6;
    --trace:#0E7C86; --trace-soft:#D7EBED;
    --good:#2C7A57; --good-soft:#DCEDE4;
    --warn:#9A6612; --warn-soft:#F3E8D2;
    --bad:#AC3B3E; --bad-soft:#F5DFDF;
    --focus:#0E7C86;
    --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  }
  @media (prefers-color-scheme:dark){
    :root:not([data-theme="light"]){
      --ground:#0C1114; --panel:#131B1E; --panel-2:#182226; --sunk:#0A0F11;
      --ink:#E3EDED; --ink-2:#A2B4B6; --ink-3:#71858A;
      --rule:#222F33; --rule-2:#314247;
      --trace:#45BFC8; --trace-soft:#12333A;
      --good:#5FBE90; --good-soft:#122B21;
      --warn:#D2A257; --warn-soft:#2C2413;
      --bad:#E1797C; --bad-soft:#2E1719;
      --focus:#45BFC8;
    }
  }
  :root[data-theme="dark"]{
    --ground:#0C1114; --panel:#131B1E; --panel-2:#182226; --sunk:#0A0F11;
    --ink:#E3EDED; --ink-2:#A2B4B6; --ink-3:#71858A;
    --rule:#222F33; --rule-2:#314247;
    --trace:#45BFC8; --trace-soft:#12333A;
    --good:#5FBE90; --good-soft:#122B21;
    --warn:#D2A257; --warn-soft:#2C2413;
    --bad:#E1797C; --bad-soft:#2E1719;
    --focus:#45BFC8;
  }

  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);
       font-family:var(--mono);font-size:14px;line-height:1.5;
       -webkit-font-smoothing:antialiased}
  .prose{font-family:var(--sans)}
  :focus-visible{outline:2px solid var(--focus);outline-offset:2px}

  .shell{max-width:1240px;margin:0 auto;padding:28px 22px 72px;
         display:flex;flex-direction:column;gap:22px}

  /* ---- header ---- */
  header{display:flex;flex-direction:column;gap:12px}
  .brandline{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
  .brand{font-size:12px;letter-spacing:.16em;text-transform:uppercase;
         color:var(--trace);font-weight:600}
  .brand-sub{font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;
             color:var(--ink-3)}
  h1{font-family:var(--sans);font-size:clamp(25px,3.6vw,36px);line-height:1.12;
     letter-spacing:-.022em;font-weight:680;margin:0;max-width:22ch;
     text-wrap:balance}
  .thesis{font-family:var(--sans);font-size:16px;color:var(--ink-2);
          max-width:66ch;margin:0}
  .thesis b{color:var(--ink);font-weight:620}
  /* Provenance is not a footnote. This page is built to be screenshotted, and
     a screenshot of invented quality numbers is the single worst artefact this
     project could produce — so when the scores are synthetic, the page says so
     above the fold, in the warning colour, before anything else is read. */
  .prov{display:flex;gap:11px;align-items:flex-start;padding:11px 13px;
        border-radius:3px;font-size:12.5px;line-height:1.5;border:1px solid var(--rule-2);
        background:var(--panel);color:var(--ink-2)}
  .prov .dot{width:7px;height:7px;border-radius:50%;background:currentColor;
             flex:none;margin-top:6px}
  .prov b{color:var(--ink);font-weight:650}
  .prov.mock{border-color:var(--warn);background:var(--warn-soft);color:var(--warn)}
  .prov.mock b{color:var(--warn)}
  .prov.real{border-color:var(--good);background:var(--good-soft);color:var(--good)}
  .prov.real b{color:var(--good)}
  .coverage{display:flex;gap:8px;flex-wrap:wrap;margin-top:2px}
  .stat{background:var(--panel);border:1px solid var(--rule);border-radius:3px;
        padding:5px 10px;font-size:11.5px;color:var(--ink-2);
        font-variant-numeric:tabular-nums}
  .stat b{color:var(--ink);font-weight:620}

  /* ---- search bar: the instrument's front panel ---- */
  .console{background:var(--panel);border:1px solid var(--rule-2);border-radius:4px;
           box-shadow:0 1px 0 var(--rule);overflow:hidden}
  .bar{display:flex;align-items:center;gap:10px;padding:0 14px;
       border-bottom:1px solid var(--rule);background:var(--panel)}
  .bar .caret{color:var(--trace);font-size:16px;font-weight:700;flex:none}
  #q{flex:1;border:0;background:transparent;color:var(--ink);
     font-family:var(--mono);font-size:16px;padding:16px 0;outline:none;
     min-width:0}
  #q::placeholder{color:var(--ink-3)}
  .bar .hint{font-size:11px;color:var(--ink-3);flex:none;white-space:nowrap}

  /* ---- parse readout: proof the NL parsing is real and deterministic ---- */
  .readout{display:flex;align-items:center;gap:8px;flex-wrap:wrap;
           padding:10px 14px;background:var(--panel-2);
           border-bottom:1px solid var(--rule);min-height:42px}
  .rlabel{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;
          color:var(--ink-3);margin-right:2px}
  .chip{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;
        padding:3px 9px;border-radius:2px;border:1px solid var(--rule-2);
        background:var(--panel);color:var(--ink-2)}
  .chip.term{border-color:var(--trace);color:var(--trace);background:var(--trace-soft)}
  .chip.good{border-color:var(--good);color:var(--good);background:var(--good-soft)}
  .chip.bad{border-color:var(--bad);color:var(--bad);background:var(--bad-soft)}
  .chip .kk{color:var(--ink-3);font-size:10px;letter-spacing:.08em;
            text-transform:uppercase}
  .chip.good .kk,.chip.bad .kk,.chip.term .kk{color:inherit;opacity:.72}

  /* ---- controls ---- */
  .controls{display:flex;align-items:center;gap:20px;flex-wrap:wrap;
            padding:11px 14px;background:var(--panel)}
  .ctl{display:flex;align-items:center;gap:10px;font-size:11.5px;color:var(--ink-2)}
  .ctl label{letter-spacing:.1em;text-transform:uppercase;font-size:10.5px;
             color:var(--ink-3)}
  input[type=range]{accent-color:var(--trace);width:190px}
  .wval{font-variant-numeric:tabular-nums;color:var(--ink);min-width:2.6em}
  .wnote{font-size:11px;color:var(--ink-3);flex:1;min-width:200px;text-align:right}

  /* ---- example queries ---- */
  .examples{display:flex;gap:7px;flex-wrap:wrap;align-items:center}
  .ex{font-family:var(--mono);font-size:11.5px;padding:5px 10px;border-radius:2px;
      border:1px dashed var(--rule-2);background:transparent;color:var(--ink-2);
      cursor:pointer;transition:border-color .12s,color .12s}
  .ex:hover{border-style:solid;border-color:var(--trace);color:var(--trace)}

  /* ---- results ---- */
  .split{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:18px;
         align-items:start}
  @media (max-width:940px){.split{grid-template-columns:1fr}}
  .hitcol{display:flex;flex-direction:column;gap:9px;min-width:0}
  .hits{display:flex;flex-direction:column;gap:7px;min-width:0}
  .resline{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--ink-3);
           padding:0 2px;font-variant-numeric:tabular-nums}
  .resline b{color:var(--ink-2);font-weight:600}
  .resline .warnq{color:var(--warn)}
  .empty{padding:34px 18px;text-align:center;color:var(--ink-3);
         background:var(--panel);border:1px solid var(--rule);border-radius:4px}

  .hit{display:grid;grid-template-columns:34px minmax(0,1fr) 172px;gap:14px;
       align-items:center;background:var(--panel);border:1px solid var(--rule);
       border-left:2px solid var(--rule-2);border-radius:3px;padding:11px 13px;
       cursor:pointer;text-align:left;width:100%;font:inherit;color:inherit;
       transition:border-color .12s,background .12s}
  .hit:hover{border-color:var(--rule-2);background:var(--panel-2)}
  .hit[aria-current="true"]{border-left-color:var(--trace);background:var(--panel-2)}
  .hit.clean{border-left-color:var(--good)}
  .hit.dirty{border-left-color:var(--bad)}
  .hit.unaudited{border-left-style:dashed}
  .rank{font-size:15px;color:var(--ink-3);font-variant-numeric:tabular-nums;
        text-align:right}
  .hit[aria-current="true"] .rank{color:var(--trace)}
  .hbody{min-width:0;display:flex;flex-direction:column;gap:4px}
  .task{font-size:14px;color:var(--ink);overflow:hidden;text-overflow:ellipsis;
        white-space:nowrap}
  .meta{font-size:11px;color:var(--ink-3);display:flex;gap:9px;flex-wrap:wrap;
        font-variant-numeric:tabular-nums}
  .meta .flagbad{color:var(--bad)}
  .meta .flagnone{color:var(--ink-3);font-style:italic}
  .meters{display:flex;flex-direction:column;gap:3px}
  .meter{display:grid;grid-template-columns:52px 1fr 30px;gap:7px;
         align-items:center;font-size:10px;color:var(--ink-3)}
  /* both are spans: .track is blockified by the grid, .fill is not — without
     display:block an inline span ignores width and height and every bar
     renders empty */
  .track{display:block;height:5px;background:var(--sunk);border-radius:1px;
         overflow:hidden}
  .fill{display:block;height:100%;width:0;background:var(--ink-3);
        border-radius:1px;transition:width .18s ease}
  .fill.sem{background:var(--trace)}
  .fill.sig{background:var(--ink-2)}
  .fill.suc{background:var(--good)}
  .fill.suc.low{background:var(--bad)}
  .mval{font-variant-numeric:tabular-nums;color:var(--ink-2);text-align:right}
  .mval.na{color:var(--ink-3)}

  /* ---- detail panel ---- */
  .detail{position:sticky;top:18px;background:var(--panel);
          border:1px solid var(--rule);border-radius:4px;overflow:hidden}
  .dhead{padding:13px 15px;border-bottom:1px solid var(--rule);
         background:var(--panel-2);display:flex;flex-direction:column;gap:4px}
  .dtask{font-size:13.5px;word-break:break-word}
  .did{font-size:10.5px;color:var(--ink-3);word-break:break-all}
  .dbody{padding:15px;display:flex;flex-direction:column;gap:16px}
  .verdict{display:flex;align-items:center;gap:9px;font-size:12px;padding:8px 11px;
           border-radius:2px;border:1px solid var(--rule-2);background:var(--panel-2)}
  .verdict.ok{border-color:var(--good);color:var(--good);background:var(--good-soft)}
  .verdict.no{border-color:var(--bad);color:var(--bad);background:var(--bad-soft)}
  .verdict .dot{width:7px;height:7px;border-radius:50%;background:currentColor;flex:none}
  .dsec{display:flex;flex-direction:column;gap:8px}
  .dsec h3{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;
           color:var(--ink-3);font-weight:600;margin:0}
  .chan{display:grid;grid-template-columns:1fr auto;gap:3px 10px;font-size:11.5px;
        color:var(--ink-2);font-variant-numeric:tabular-nums}
  .chan b{color:var(--ink);font-weight:600}
  .chan .na{color:var(--ink-3)}
  .eyewrap{display:flex;gap:13px;align-items:center}
  .eyecap{font-size:10.5px;color:var(--ink-3);line-height:1.45}
  .timeline{position:relative;height:26px;background:var(--sunk);border-radius:2px;
            border:1px solid var(--rule)}
  .tick{position:absolute;top:0;bottom:0;width:2px;background:var(--bad);
        transform:translateX(-1px)}
  .tlabels{display:flex;justify-content:space-between;font-size:10px;
           color:var(--ink-3);font-variant-numeric:tabular-nums}

  /* ---- footer ---- */
  footer{display:flex;flex-direction:column;gap:10px;padding-top:16px;
         border-top:1px solid var(--rule)}
  .method{font-family:var(--sans);font-size:13px;color:var(--ink-2);max-width:80ch}
  .method b{color:var(--ink);font-weight:620}
  .badges{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .badge{font-size:11px;padding:4px 9px;border-radius:2px;border:1px solid var(--rule-2);
         color:var(--ink-2);background:var(--panel)}
  .badge.pass{border-color:var(--good);color:var(--good);background:var(--good-soft)}
  .badge.fail{border-color:var(--bad);color:var(--bad);background:var(--bad-soft)}
  @media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="shell">
  <header>
    <div class="brandline">
      <span class="brand">egoeye</span>
      <span class="brand-sub">quality-aware episode search · EgoVerse track 3</span>
    </div>
    <h1>Find the demonstration, then find the good one.</h1>
    <p class="thesis prose">A text index answers <b>what an episode is of</b>. The egoeye
      audit answers <b>whether the human did it cleanly</b> — deterministically, from wrist
      kinematics, no judge model. Neither is a training-set filter alone. Joined, they
      answer the query a robot-learning team actually types.</p>
    <div class="coverage" id="coverage"></div>
    <div id="provenance"></div>
  </header>

  <div class="console">
    <div class="bar">
      <span class="caret" aria-hidden="true">&rsaquo;</span>
      <input id="q" type="text" autocomplete="off" spellcheck="false"
             aria-label="Search episodes in natural language"
             placeholder="clean examples of washing dishes">
      <span class="hint">438k episodes · rule-based parse</span>
    </div>
    <div class="readout" id="readout"></div>
    <div class="controls">
      <div class="ctl">
        <label for="w">quality weight</label>
        <input id="w" type="range" min="0" max="1.5" step="0.05">
        <span class="wval" id="wval"></span>
      </div>
      <span class="wnote" id="wnote"></span>
    </div>
  </div>

  <div class="examples" id="examples"></div>

  <div class="split">
    <div class="hitcol">
      <div class="resline" id="resline"></div>
      <div class="hits" id="hits"></div>
    </div>
    <aside class="detail" id="detail"></aside>
  </div>

  <footer>
    <p class="method prose"><b>How the ranking works.</b> final = semantic ×
      quality<sup>w</sup>. Semantic is BM25 over task text with domain synonym
      expansion. Quality blends <b>success</b> (1 − failure_score: discrete drop and
      collision events) with <b>signal</b> (eye opening, rainflow retry ratio, mask
      violations, tracking dropout). Multiplicative, so a pristine but irrelevant
      episode can never surface — quality re-orders relevance, it never replaces it.
      Set w to 0 for the ablation: plain semantic search.</p>
    <div class="badges" id="badges"></div>
  </footer>
</div>

<script>
const DATA = __PAYLOAD__;
const LEX = DATA.lex;
const S = new Set(LEX.stop), SCAFFOLD = new Set(LEX.scaffold);
const SUCC = new Set(LEX.success), FAIL = new Set(LEX.failure);
const SUCC_ST = new Set(LEX.successStems), FAIL_ST = new Set(LEX.failureStems);
const NEG = new Set(LEX.neg), AMBIG = new Set(LEX.ambiguous);
const LABS = new Set(LEX.labs), PREPS = new Set(LEX.labPreps);

/* ---- transliteration of egosearch.stem / tokenize ---- */
function stem(w){
  if(w.length<=3) return w;
  const rules=[["ies",3],["ing",3],["ed",2],["es",2],["s",1]];
  for(const [suf,keep] of rules){
    if(w.endsWith(suf) && w.length-keep>=3){
      let base=w.slice(0,-keep);
      if(suf==="ies") return base+"y";
      if((suf==="ing"||suf==="ed") && base.length>3 &&
         base[base.length-1]===base[base.length-2]) base=base.slice(0,-1);
      return base;
    }
  }
  return w;
}
const split = t => String(t).toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
function tokenize(text){
  return split(text).filter(t=>!S.has(t) && !/^\d+$/.test(t)).map(stem);
}

/* ---- transliteration of egosearch.parse_query ---- */
const UNIT={s:1,sec:1,secs:1,second:1,seconds:1,m:60,min:60,mins:60,minute:60,minutes:60};
const NUMUNIT="(\\d+(?:\\.\\d+)?)\\s*(s|secs?|seconds?|m|mins?|minutes?)\\b";
const UNDER="(?:under|below|less than|shorter than|at most|<=?|max)";
const OVER="(?:over|above|more than|longer than|at least|>=?|min)";
function parseQuery(raw){
  const q={raw:raw,terms:[],rawTerms:[],intent:"none",evidence:[],lab:null,
           embodiment:null,minDur:null,maxDur:null,k:null};
  let text=" "+raw.toLowerCase().replace(/[’']/g,"")+" ";
  let m=text.match(/\b(?:lab|source)\s*:\s*([a-z0-9_]+)/);
  if(m) q.lab=m[1];
  text=text.replace(/\b(?:lab|source)\s*:\s*[a-z0-9_]+/g," ");

  m=text.match(new RegExp(UNDER+"\\s*"+NUMUNIT));
  if(m) q.maxDur=parseFloat(m[1])*UNIT[m[2]];
  m=text.match(new RegExp(OVER+"\\s*"+NUMUNIT));
  if(m) q.minDur=parseFloat(m[1])*UNIT[m[2]];
  if(q.maxDur===null && q.minDur===null){
    if(/\b(short|brief|quick)\b/.test(text)) q.maxDur=15;
    else if(/\b(long|lengthy|extended)\b/.test(text)) q.minDur=60;
  }
  text=text.replace(new RegExp(UNDER+"\\s*"+NUMUNIT,"g")," ")
           .replace(new RegExp(OVER+"\\s*"+NUMUNIT,"g")," ")
           .replace(/\b(short|brief|quick|long|lengthy|extended)\b/g," ");
  m=text.match(/\b(?:top|first|best|worst)\s+(\d{1,3})\b/);
  if(m) q.k=parseInt(m[1],10);

  const EMB=[[/\b(bimanual|two[- ]hand(ed)?|both hands)\b/,"bimanual"],
             [/\b(left[- ](hand|arm)(ed)?|one[- ]hand(ed)? left)\b/,"left"],
             [/\b(right[- ](hand|arm)(ed)?)\b/,"right"],
             [/\b(robot|eva|yam|teleop)\b/,"robot"]];
  for(const [re,name] of EMB){
    if(re.test(text)){ q.embodiment=name; text=text.replace(re," "); break; }
  }

  const words=split(text);
  if(q.lab===null){
    for(let i=1;i<words.length;i++){
      if(LABS.has(words[i]) && PREPS.has(words[i-1])){ q.lab=words[i]; words[i]=""; break; }
    }
  }
  const pos=[],neg=[],consumed=new Set();
  for(let i=0;i<words.length;i++){
    const w=words[i]; if(!w) continue;
    const st=stem(w);
    let isS=SUCC.has(w)||SUCC_ST.has(st), isF=FAIL.has(w)||FAIL_ST.has(st);
    if(AMBIG.has(w) && !words.slice(i+1,i+3).some(x=>SCAFFOLD.has(x))){ isS=false; isF=false; }
    if(!isS && !isF) continue;
    consumed.add(i);
    const negated=words.slice(Math.max(0,i-2),i).some(p=>NEG.has(p));
    if(isS){ (negated?neg:pos).push(w); } else { (negated?pos:neg).push(w); }
  }
  if(pos.length && !neg.length){ q.intent="success"; q.evidence=pos; }
  else if(neg.length && !pos.length){ q.intent="failure"; q.evidence=neg; }
  else if(pos.length && neg.length){
    q.intent = pos.length>=neg.length ? "success":"failure"; q.evidence=pos.concat(neg);
  }
  const drop=new Set([...NEG,"top","first","best","worst"]);
  q.rawTerms=words.filter((w,i)=>w && !consumed.has(i) && !S.has(w) && !drop.has(w)
                                  && !/^\d+$/.test(w));
  q.terms=q.rawTerms.map(stem);
  return q;
}

/* ---- index: BM25 over the distinct document texts ---- */
const NDOC=DATA.docs.length;
const postings=new Map(), docLen=new Float32Array(NDOC);
DATA.docs.forEach((text,i)=>{
  const toks=tokenize(text); docLen[i]=toks.length||1;
  for(const t of toks){
    let p=postings.get(t); if(!p){ p=new Map(); postings.set(t,p); }
    p.set(i,(p.get(i)||0)+1);
  }
});
const avgdl=docLen.reduce((a,b)=>a+b,0)/(NDOC||1);
const idf=new Map();
for(const [t,p] of postings) idf.set(t, Math.log(1+(NDOC-p.size+0.5)/(p.size+0.5)));

function scoreDocs(terms){
  const out=new Float32Array(NDOC);
  if(!terms.length) return out;
  let ideal=0;
  for(const term of terms){
    const variants=[[term,1]].concat((LEX.syn[term]||[]).map(s=>[s,LEX.synWeight]));
    let base=idf.get(term);
    if(base===undefined){
      base=0;
      for(const [v,w] of variants) base=Math.max(base,(idf.get(v)||0)*w);
    }
    ideal+=base;
    const ts=new Float32Array(NDOC);
    for(const [variant,weight] of variants){
      const p=postings.get(variant); if(!p) continue;
      const w=idf.get(variant)*weight;
      for(const [d,tf] of p){
        const denom=tf+LEX.k1*(1-LEX.b+LEX.b*docLen[d]/avgdl);
        const c=w*tf*(LEX.k1+1)/denom;
        if(c>ts[d]) ts[d]=c;
      }
    }
    for(let i=0;i<NDOC;i++) out[i]+=ts[i];
  }
  if(ideal<=0) return out;
  for(let i=0;i<NDOC;i++) out[i]/=ideal;
  if(terms.length>1){
    const cover=new Float32Array(NDOC);
    for(const term of terms){
      const seen=new Uint8Array(NDOC);
      const variants=[term].concat(LEX.syn[term]||[]);
      for(const v of variants){ const p=postings.get(v); if(p) for(const d of p.keys()) seen[d]=1; }
      for(let i=0;i<NDOC;i++) cover[i]+=seen[i];
    }
    for(let i=0;i<NDOC;i++) out[i]*=(0.35+0.65*cover[i]/terms.length);
  }
  for(let i=0;i<NDOC;i++) out[i]=Math.min(1,Math.max(0,out[i]));
  return out;
}

/* ---- episodes ---- */
const clamp=(v,lo=0,hi=1)=>Math.min(hi,Math.max(lo,v));
const EPS=[];
for(const r of DATA.lite)
  EPS.push({id:r[0],doc:r[1],lab:DATA.labs[r[2]],emb:DATA.embs[r[3]],dur:r[4],
            task:r[5],scored:false,sig:null,sigRaw:null,suc:null,imp:[]});
for(const r of DATA.full){
  // signal and success both arrive precomputed: one is a percentile within the
  // audited corpus, the other is normalised against a corpus-derived impulse
  // rate — neither is reconstructible from the subset shipped to the browser
  EPS.push({id:r[0],doc:r[1],lab:DATA.labs[r[2]],emb:DATA.embs[r[3]],dur:r[4],
            task:r[5],scored:true,sig:r[6],sigRaw:r[7],suc:r[8],sucRaw:r[9],
            fs:r[10],eye:r[11],rf:r[12],mask:r[13],nan:r[14],nimp:r[15],
            imp:r[16]||[]});
}

function fuse(sem,sig,suc,intent,w){
  if(intent==="none"||w<=0) return sem;
  const g=sig===null?0.5:sig, c=suc===null?0.5:suc;
  const q = intent==="success" ? 0.65*c+0.35*g : 0.65*(1-c)+0.35*(1-g);
  return sem*Math.pow(clamp(q,1e-6,1),w);
}

function search(raw,k,w){
  const q=parseQuery(raw), sem=scoreDocs(q.terms);
  const out=[];
  for(const e of EPS){
    if(q.lab && e.lab.toLowerCase()!==q.lab) continue;
    if(q.embodiment){
      const em=e.emb.toLowerCase();
      if(q.embodiment==="robot"){ if(em.startsWith("human")) continue; }
      else if(!em.includes(q.embodiment)) continue;
    }
    if(q.minDur!==null && !(e.dur>=q.minDur)) continue;
    if(q.maxDur!==null && !(e.dur!==null && e.dur<=q.maxDur)) continue;
    const s=sem[e.doc];
    if(q.terms.length && !(s>0)) continue;
    out.push([fuse(s,e.sig,e.suc,q.intent,w),s,e]);
  }
  // same tiebreak as the Python lexsort: score desc, then episode_id asc, so
  // the two engines cannot disagree just because they built their pools in a
  // different order
  out.sort((a,b)=> (b[0]-a[0]) || (a[2].id<b[2].id?-1:(a[2].id>b[2].id?1:0)));
  return {q:q,hits:out.slice(0,q.k||k)};
}

/* ---- rendering ---- */
const $=s=>document.querySelector(s);
const fmt=v=>(v===null||v===undefined||Number.isNaN(v))?"—":v.toFixed(2);
const pct=v=>(v===null||v===undefined||Number.isNaN(v))?0:clamp(v)*100;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

let W=DATA.qw, selected=null, current=null;

function meter(cls,label,v){
  const na=(v===null||v===undefined||Number.isNaN(v));
  const low=(cls==="suc"&&!na&&v<0.5)?" low":"";
  return `<div class="meter"><span>${label}</span>
    <span class="track"><span class="fill ${cls}${low}" style="width:${pct(v)}%"></span></span>
    <span class="mval${na?" na":""}">${fmt(v)}</span></div>`;
}

function eyeGlyph(open,mask){
  // Schematic, not a raw overlay: aperture height is eye_opening, edge scatter
  // is mask_violation_p90. Both come straight from the audit row.
  // Colours go through style="", never fill="var(--x)": var() does not
  // substitute inside SVG presentation attributes, so the attribute form
  // silently renders black on both themes.
  if(open===null||open===undefined||Number.isNaN(open))
    return `<svg width="86" height="52" viewBox="0 0 86 52" role="img"
      aria-label="no eye diagram for this episode">
      <rect x="1" y="1" width="84" height="50" style="fill:none;stroke:var(--rule)"/>
      <text x="43" y="30" text-anchor="middle"
        style="font-size:9px;fill:var(--ink-3);font-family:var(--mono)">no eye</text></svg>`;
  const h=6+34*clamp(open), top=26-h/2, bot=26+h/2;
  const j=clamp(mask===null||mask===undefined?0.12:mask,0,0.4)*26;
  let lines="";
  for(let i=0;i<7;i++){
    const o=(i-3)/3*j, op=(0.28+0.1*(3-Math.abs(i-3))).toFixed(2);
    const st=`fill:none;stroke:var(--trace);stroke-width:1;opacity:${op}`;
    lines+=`<path d="M6 26 C 24 ${top+o}, 62 ${top+o}, 80 26" style="${st}"/>
            <path d="M6 26 C 24 ${bot-o}, 62 ${bot-o}, 80 26" style="${st}"/>`;
  }
  return `<svg width="86" height="52" viewBox="0 0 86 52" role="img"
    aria-label="eye diagram, opening ${fmt(open)}">
    <rect x="1" y="1" width="84" height="50" style="fill:var(--sunk);stroke:var(--rule)"/>
    ${lines}</svg>`;
}

function renderReadout(q){
  const bits=[`<span class="rlabel">parsed</span>`];
  if(q.rawTerms.length)
    bits.push(...q.rawTerms.map(t=>`<span class="chip term">${esc(t)}</span>`));
  else bits.push(`<span class="chip">no content terms</span>`);
  if(q.intent!=="none"){
    const cls=q.intent==="success"?"good":"bad";
    bits.push(`<span class="chip ${cls}"><span class="kk">want</span>${q.intent}
      <span class="kk">via ${esc(q.evidence.join(", "))}</span></span>`);
  } else {
    bits.push(`<span class="chip"><span class="kk">quality</span>not requested</span>`);
  }
  if(q.lab) bits.push(`<span class="chip"><span class="kk">lab</span>${esc(q.lab)}</span>`);
  if(q.embodiment) bits.push(`<span class="chip"><span class="kk">hands</span>${esc(q.embodiment)}</span>`);
  if(q.maxDur!==null) bits.push(`<span class="chip"><span class="kk">max</span>${q.maxDur}s</span>`);
  if(q.minDur!==null) bits.push(`<span class="chip"><span class="kk">min</span>${q.minDur}s</span>`);
  $("#readout").innerHTML=bits.join("");
}

function renderResLine(res){
  const n=res.hits.length, aud=res.hits.filter(h=>h[2].scored).length;
  const bits=[`<span><b>${n}</b> shown</span>`,
              `<span><b>${aud}</b> with quality evidence</span>`];
  if(res.q.intent==="none")
    bits.push(`<span>ranking on relevance only — no quality intent in the query</span>`);
  else if(aud===0)
    bits.push(`<span class="warnq">the audit has not reached this task family yet;
      every hit is ranked on text alone</span>`);
  else
    bits.push(`<span>ordered by ${res.q.intent==="success"?"cleanest":"most-fumbled"} first</span>`);
  $("#resline").innerHTML=bits.join("");
}

function renderHits(res){
  const el=$("#hits");
  renderResLine(res);
  if(!res.hits.length){
    el.innerHTML=`<div class="empty">No episode matches those terms.<br>
      Try <code>fold clothes</code>, <code>pack groceries</code> or <code>wash dishes</code>.</div>`;
    $("#detail").innerHTML=""; return;
  }
  el.innerHTML=res.hits.map(([score,sem,e],i)=>{
    const cls = !e.scored ? "unaudited" : (e.suc>=0.7 ? "clean" : (e.suc<0.4?"dirty":""));
    const dur = e.dur===null?"—":e.dur.toFixed(1)+"s";
    const flag = !e.scored ? `<span class="flagnone">unaudited</span>`
      : (e.nimp>0 ? `<span class="flagbad">${e.nimp} impulse${e.nimp>1?"s":""}</span>`
                  : `<span>no impulse</span>`);
    return `<button class="hit ${cls}" data-i="${i}" aria-current="${i===selected}">
      <span class="rank">${i+1}</span>
      <span class="hbody">
        <span class="task">${esc(e.task)}</span>
        <span class="meta"><span>${esc(e.lab)}</span><span>${dur}</span>${flag}
          <span>final ${score.toFixed(3)}</span></span>
      </span>
      <span class="meters">
        ${meter("sem","semantic",sem)}
        ${meter("sig","signal",e.sig)}
        ${meter("suc","success",e.suc)}
      </span></button>`;
  }).join("");
  el.querySelectorAll(".hit").forEach(b=>b.addEventListener("click",()=>{
    selected=+b.dataset.i; renderHits(current); renderDetail(current.hits[selected]);
  }));
  renderDetail(res.hits[Math.min(selected||0,res.hits.length-1)]);
}

function renderDetail(hit){
  if(!hit){ $("#detail").innerHTML=""; return; }
  const [score,sem,e]=hit;
  const dur=e.dur||1;
  const ticks=(e.imp||[]).map(t=>
    `<span class="tick" style="left:${clamp(t/dur)*100}%" title="${t}s"></span>`).join("");
  const verdict = !e.scored
    ? `<div class="verdict"><span class="dot"></span>Not yet audited — no quality evidence</div>`
    : (e.suc>=0.6
       ? `<div class="verdict ok"><span class="dot"></span>No failure event detected</div>`
       : `<div class="verdict no"><span class="dot"></span>Probable failure event</div>`);
  const chan=(label,v,note)=>
    `<span>${label}</span><span class="${v===null||v===undefined?"na":""}"><b>${fmt(v)}</b>
      ${note?`<span class="na"> ${note}</span>`:""}</span>`;
  $("#detail").innerHTML=`
    <div class="dhead"><span class="dtask">${esc(e.task)}</span>
      <span class="did">${esc(e.id)}</span>
      <span class="did">${esc(e.lab)} · ${esc(e.emb)} · ${e.dur===null?"—":e.dur.toFixed(1)+"s"}</span>
    </div>
    <div class="dbody">
      ${verdict}
      <div class="dsec"><h3>ranking</h3>
        <div class="chan">
          ${chan("semantic",sem)}${chan("signal",e.sig)}${chan("success",e.suc)}
          ${chan("final",score,"= sem × q^"+W.toFixed(2))}
        </div></div>
      <div class="dsec"><h3>eye diagram</h3>
        <div class="eyewrap">${eyeGlyph(e.eye,e.mask)}
          <span class="eyecap">aperture = eye_opening<br>edge scatter = mask violations<br>
            <span style="opacity:.75">schematic from the two audit numbers</span></span>
        </div></div>
      <div class="dsec"><h3>signal channels</h3>
        <div class="chan">
          ${chan("eye opening",e.eye)}${chan("rainflow small-cycle",e.rf)}
          ${chan("mask violation p90",e.mask)}${chan("tracking NaN",e.nan)}
          ${chan("raw composite",e.sigRaw)}
        </div>
        <span class="eyecap">signal is the raw composite's percentile within the
          audited corpus — the absolute mappings saturate on real data.</span>
      </div>
      <div class="dsec"><h3>failure evidence</h3>
        <div class="chan">
          ${chan("impulses",e.nimp===null?null:e.nimp)}
          ${chan("per minute",(e.nimp===null||!e.dur)?null:e.nimp/(e.dur/60))}
          ${chan("detector failure_score",e.fs)}
          ${chan("success (1 - that)",e.sucRaw)}
        </div>
        <span class="eyecap">${DATA.successScale==="rate"
          ? `success is ranked on impulses per minute against a reference of
             ${DATA.rateRef} /min — the raw count is partly a measure of episode
             length (corr ${DATA.durCorr} here), so ranking on it sorts by duration.`
          : `success is the detector's raw score; the impulse count it is built on
             carries a length bias.`}</span>
      </div>
      <div class="dsec"><h3>impulse events</h3>
        <div class="timeline">${ticks}</div>
        <div class="tlabels"><span>0s</span>
          <span>${(e.imp&&e.imp.length)?e.imp.length+" event"+(e.imp.length>1?"s":""):"none detected"}</span>
          <span>${e.dur===null?"—":e.dur.toFixed(1)+"s"}</span></div>
      </div>
    </div>`;
}

function run(raw,resetSel){
  if(resetSel!==false) selected=0;
  current=search(raw,10,W);
  renderReadout(current.q);
  renderHits(current);
}

/* ---- wiring ---- */
const stats=DATA.stats;
$("#coverage").innerHTML=[
  ["episodes indexed",stats.episodes_total.toLocaleString()],
  ["distinct task texts",stats.docs_total.toLocaleString()],
  ["audited by egoeye",stats.audited.toLocaleString()],
  ["shipped in this page",stats.shipped.toLocaleString()],
].map(([k,v])=>`<span class="stat">${k} <b>${v}</b></span>`).join("");

const prov=$("#provenance");
if(DATA.synthetic){
  prov.className="prov mock";
  prov.innerHTML=`<span class="dot"></span><span><b>The quality numbers on this page are
    invented.</b> Retrieval runs on the real ${stats.episodes_total.toLocaleString()}-episode
    catalogue, but signal and success come from <code>${esc(DATA.source)}</code> — a synthetic
    parquet built to develop against while the fan-out ran. Nothing here is a measurement.
    Rebuild with <code>--results audit_results.parquet</code> before this is shown to anyone.</span>`;
} else {
  prov.className="prov real";
  prov.innerHTML=`<span class="dot"></span><span><b>Real scores.</b> Signal and success come
    from <code>${esc(DATA.source)}</code> — ${stats.audited.toLocaleString()} episodes measured
    on Modal from their R2 keypoints. The remaining
    ${(stats.episodes_total-stats.audited).toLocaleString()} are indexed for retrieval and
    marked <b>unaudited</b>: they carry no quality evidence and the ranking does not pretend
    otherwise.</span>`;
}

$("#examples").innerHTML=DATA.reference.map(r=>
  `<button class="ex" data-q="${esc(r.q)}">${esc(r.q)}</button>`).join("");
$("#examples").querySelectorAll(".ex").forEach(b=>b.addEventListener("click",()=>{
  $("#q").value=b.dataset.q; run(b.dataset.q);
}));

const wEl=$("#w");
wEl.value=W;
function syncW(){
  $("#wval").textContent="w = "+W.toFixed(2);
  $("#wnote").textContent = W<=0
    ? "ablation: pure semantic search, quality ignored"
    : (W<0.5 ? "relevance dominates; quality breaks ties"
             : (W<=1 ? "balanced — the default" : "quality dominates within relevance"));
}
wEl.addEventListener("input",()=>{ W=parseFloat(wEl.value); syncW();
  run($("#q").value||$("#q").placeholder,false); });
$("#q").addEventListener("input",()=>run($("#q").value||$("#q").placeholder));
syncW();
// open on the first canned query, which is guaranteed to have audit coverage
if(DATA.reference.length) $("#q").placeholder=DATA.reference[0].q;
run($("#q").placeholder);

/* ---- engine parity: does the browser reproduce the CLI? ---- */
(function parity(){
  let match=0;
  for(const ref of DATA.reference){
    const got=search(ref.q,5,DATA.qw).hits.slice(0,5).map(h=>h[2].id);
    if(JSON.stringify(got)===JSON.stringify(ref.top)) match++;
  }
  const n=DATA.reference.length, ok=(match===n);
  $("#badges").innerHTML=
    `<span class="badge ${ok?"pass":"fail"}">engine parity ${match}/${n} queries match the Python CLI</span>`+
    `<span class="badge">deterministic · no model, no network</span>`+
    (DATA.source?`<span class="badge">scores: ${esc(DATA.source)}</span>`:"");
})();
</script>
"""
