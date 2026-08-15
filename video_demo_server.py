"""
Live, video-forward version of the search demo — for presenting, not reading.

demo_search.html is a static, shareable page (client-side re-run for engine
parity, ~4-45MB depending on scope). This is the opposite trade: a small
local Flask server that streams the *real* preview MP4 from R2 on click and
draws the *real* confidence trace synced to video playback, so a judge can
watch the score dip exactly when the hand fumbles. Costs a live R2
connection; buys real video for as many results as you want (default 50,
not demo_search.html's static top-10 gallery).

Fully server-rendered, zero client-side fetch(): search and episode
selection are plain GET navigations (`/?q=...&episode=...`), same as typing
a URL in the address bar. This was a deliberate fallback after fetch()-based
XHR silently hung in one browser environment (likely blocked by local
security software) while direct navigation worked fine every time. The only
JS left is canvas drawing + video timeupdate sync, both fully local — the
confidence trace ships as inline JSON in the page, not a separate request.

Usage:
  set -a; . ~/.egoverse_env; set +a
  python video_demo_server.py --episodes episodes_washdishes.csv \\
      --results audit_washdishes_local.parquet
  # open http://127.0.0.1:8080

First video click per episode downloads+caches to .video_cache/ (a few
hundred ms to a few seconds depending on clip length); repeat clicks are
instant (native <video> requests, not JS fetch — browsers handle these
through the normal resource-loading path, not XHR/fetch).
"""

from __future__ import annotations

import argparse
import html as htmlmod
import json
import os
import sys
import threading

import numpy as np
import pandas as pd
from flask import Flask, Response, request, send_file

sys.path.insert(0, os.path.expanduser("~/EgoVerse"))

import egosearch as E
from eyekit import (clean_trajectory, confidence_trace, eye_matrix,
                    eye_metrics, kinematics, segment_cycles)
from make_demo_figs import load as load_wrist

app = Flask(__name__)

STATE: dict = {}
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".video_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
_trace_cache: dict[str, dict] = {}
_trace_lock = threading.Lock()
DEFAULT_Q = "wash dishes cleanly, no drops"


def _r2_client():
    import boto3
    endpoint = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
    return boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


def _analysis_for(episode_id: str) -> dict:
    """Confidence trace + eye diagram, computed from one zarr load (R2 fetch
    is the expensive part — don't pay for it twice for two charts)."""
    with _trace_lock:
        cached = _trace_cache.get(episode_id)
    if cached is not None:
        return cached

    zarr_path = STATE["zarr_lookup"].get(episode_id)
    if not zarr_path:
        return {"error": "no zarr_path for this episode"}

    fps = float(STATE["fps_lookup"].get(episode_id, 30.0))
    xyz = load_wrist(zarr_path, None, 0)

    t, conf = confidence_trace(xyz, fps)
    dips = []
    below = conf < 0.9
    i = 0
    while i < len(below):
        if below[i]:
            j = i
            while j < len(below) and below[j]:
                j += 1
            k = i + int(np.argmin(conf[i:j]))
            dips.append({"t": float(t[k]), "conf": float(conf[k])})
            i = j
        else:
            i += 1

    speed, _ = kinematics(clean_trajectory(xyz), fps)
    segs = segment_cycles(speed, fps)
    M = eye_matrix(speed, segs)
    if M is not None and M.shape[0] >= 1:
        eye = {
            "cycles": M.round(3).tolist(),
            "median": np.median(M, axis=0).round(3).tolist(),
        }
        if M.shape[0] >= 3:
            eye.update({k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in eye_metrics(M).items()})
    else:
        eye = None

    payload = {
        "t": t.round(2).tolist(), "conf": conf.round(3).tolist(), "dips": dips,
        "eye": eye,
    }
    with _trace_lock:
        _trace_cache[episode_id] = payload
    return payload


@app.route("/api/video/<episode_id>")
def api_video(episode_id):
    mp4_path = STATE["mp4_lookup"].get(episode_id)
    if not mp4_path:
        return "no preview_mp4 for this episode", 404

    local_path = os.path.join(CACHE_DIR, f"{episode_id}.mp4")
    if not os.path.exists(local_path):
        bucket, key = mp4_path.replace("s3://", "").split("/", 1)
        tmp = local_path + ".part"
        _r2_client().download_file(bucket, key, tmp)
        os.replace(tmp, local_path)
    return send_file(local_path, mimetype="video/mp4", conditional=True)


def esc(s) -> str:
    return htmlmod.escape(str(s if s is not None else ""))


def render_epirow(h: dict, rank: int, active: bool) -> str:
    task = esc(h["task"]).replace("_", " ")
    lab = esc(h["lab"])
    dur = f'{h["duration_s"]:.1f}s' if h.get("duration_s") else "—"
    scored = h.get("scored")
    bars = f'<div class="bar sem"><i style="width:{pct(h["semantic"])}%"></i></div>'
    if scored:
        low = " low" if h["success"] < 0.5 else ""
        bars += (f'<div class="bar sig"><i style="width:{pct(h["signal"])}%"></i></div>'
                 f'<div class="bar suc{low}"><i style="width:{pct(h["success"])}%"></i></div>')
    unaud = "" if scored else '<span class="unaud">unaudited</span>'
    cls = "epirow active" if active else "epirow"
    href = f'/?q={request.args.get("q", DEFAULT_Q)}&episode={h["episode_id"]}'
    return (f'<a class="{cls}" href="{esc(href)}">'
            f'<div class="t"><span>{task}</span><span class="rank">#{rank}</span></div>'
            f'<div class="meta"><span>{lab}</span><span>{dur}</span>{unaud}</div>'
            f'<div class="bars">{bars}</div></a>')


def pct(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return max(0.0, min(100.0, float(v) * 100))


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):.2f}"


def render_stage(h: dict) -> str:
    task = esc(h["task"]).replace("_", " ")
    meta = f'{esc(h["episode_id"])} · {esc(h["lab"])} · {esc(h["embodiment"])} · ' \
           f'{h["duration_s"]:.1f}s' if h.get("duration_s") else esc(h["episode_id"])
    scores = "".join(
        f'<div class="sc"><div class="k">{k}</div><div class="v">{fmt(v)}</div></div>'
        for k, v in [("semantic", h["semantic"]), ("signal", h.get("signal")),
                     ("success", h.get("success")), ("final", h["score"])]
    )
    if h.get("preview_mp4"):
        video_html = f'<video id="m-video" controls preload="metadata" src="/api/video/{esc(h["episode_id"])}"></video>'
    else:
        video_html = '<div class="novideo">no preview video for this episode</div>'

    analysis = _analysis_for(h["episode_id"])
    if "error" in analysis:
        dips_html = esc(analysis["error"])
        trace_json = "null"
        eye_section = ""
    else:
        n = len(analysis["dips"])
        if n:
            chips = "".join(
                f'<a class="chip" href="#" data-t="{d["t"]}">{d["t"]:.1f}s ({d["conf"]:.2f})</a>'
                for d in analysis["dips"]
            )
            dips_html = f'<b>{n} dip{"s" if n > 1 else ""}</b> detected — click a dip or the chart to jump: {chips}'
        else:
            dips_html = "no confidence dips detected in this episode"
        trace_json = json.dumps({"t": analysis["t"], "conf": analysis["conf"], "dips": analysis["dips"]})

        eye = analysis.get("eye")
        if eye is None:
            eye_section = '<div class="chartwrap"><h3>eye diagram</h3><div class="dips">not enough motion cycles detected</div></div>'
        else:
            n_cyc = len(eye["cycles"])
            opening = eye.get("eye_opening")
            cap = f'{n_cyc} cycle{"s" if n_cyc != 1 else ""}'
            if opening is not None:
                cap += f' · eye opening = {opening:.2f} (tight = consistent motion, high quality)'
            eye_section = f"""
<div class="chartwrap">
  <h3>eye diagram <span class="capnote">{cap}</span></h3>
  <canvas id="m-eye-canvas"></canvas>
</div>
<script>var EYE = {json.dumps(eye)};</script>"""

    return f"""
<div class="stagehead"><div class="task">{task}</div><div class="id">{meta}</div></div>
<div class="videowrap">{video_html}</div>
<div class="chartwrap">
  <div class="chartlegend">
    <span class="lk"><span class="dot" style="background:#4CC9E8"></span>confidence <b id="lv-conf">—</b></span>
    <span class="lk"><span class="dot" style="background:#F0616B"></span>nearest dip <b id="lv-dip">—</b></span>
  </div>
  <canvas id="m-canvas"></canvas>
</div>
<div class="dips">{dips_html}</div>
{eye_section}
<div class="scores">{scores}</div>
<div class="txt">{esc(h.get("text", ""))}</div>
<script>var TRACE = {trace_json};</script>
"""


@app.route("/")
def index():
    q = request.args.get("q", DEFAULT_Q)
    episode_id = request.args.get("episode", "")

    _, hits = STATE["search"].search(q, k=50, quality_weight=0.75)
    mp4_lookup = STATE["mp4_lookup"]
    hit_dicts = []
    for h in hits:
        d = h.to_dict()
        d["preview_mp4"] = bool(mp4_lookup.get(h.episode_id))
        hit_dicts.append(d)

    if hit_dicts:
        resline = f'{len(hit_dicts)} results for "{esc(q)}"'
        rows = "".join(render_epirow(d, d["rank"], d["episode_id"] == episode_id) for d in hit_dicts)
    else:
        resline = f'0 results for "{esc(q)}"'
        rows = '<div class="empty">No matches. Try "wash dishes" or "rinse a plate".</div>'

    selected = next((d for d in hit_dicts if d["episode_id"] == episode_id), None)
    stage = render_stage(selected) if selected else '<div class="placeholder">select an episode from the left</div>'

    chips = "".join(
        f'<a class="chip" href="/?q={esc(cq)}">{esc(label)}</a>'
        for cq, label in [
            ("wash dishes cleanly, no drops", "clean"),
            ("wash dishes that were fumbled or dropped", "fumbled"),
            ("washing a plate", "plate"),
            ("rinsing a cup", "cup"),
        ]
    )

    return Response(PAGE.format(
        q=esc(q), chips=chips, resline=resline, rows=rows, stage=stage,
    ), mimetype="text/html")


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<title>egoeye — episode viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{
  --bg:#0B1220; --panel:#111A2E; --panel-2:#0E1626; --sunk:#0A0F1C;
  --ink:#E8ECF4; --ink-2:#9AA7C2; --ink-3:#5E6B8A;
  --rule:#1E2A45; --rule-2:#283A58;
  --accent:#4CC9E8; --accent-soft:#0F2A36;
  --good:#5FD99A; --bad:#F0616B; --purple:#B39CF0;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}}
*{{box-sizing:border-box}}
html,body{{height:100%}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);overflow:hidden}}
a{{text-decoration:none;color:inherit}}
.app{{display:grid;grid-template-columns:340px 1fr;height:100%;min-height:0}}

.sidebar{{background:var(--panel);border-right:1px solid var(--rule);display:flex;flex-direction:column;min-width:0;min-height:0;height:100%}}
.sbhead{{padding:18px 18px 14px;border-bottom:1px solid var(--rule)}}
.sbhead h1{{margin:0 0 12px;font-size:18px;letter-spacing:-.01em}}
.sbhead h1 span{{color:var(--accent)}}
.searchbox{{display:flex;align-items:center;gap:8px;background:var(--sunk);border:1px solid var(--rule);
           border-radius:8px;padding:9px 12px}}
.searchbox .caret{{color:var(--accent);font-weight:700;font-family:var(--mono)}}
.searchbox input{{flex:1;border:0;background:transparent;color:var(--ink);font-size:14px;outline:none;font-family:var(--sans)}}
.chips{{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}}
.chip{{font-size:11px;padding:4px 9px;border-radius:999px;border:1px solid var(--rule);
      background:var(--panel-2);color:var(--ink-2);cursor:pointer;font-family:var(--mono);display:inline-block}}
.chip:hover{{border-color:var(--accent);color:var(--accent)}}
.resline{{padding:10px 18px;font-size:11.5px;color:var(--ink-3);font-family:var(--mono);border-bottom:1px solid var(--rule)}}

.epilist{{flex:1;overflow-y:auto;padding:6px;min-height:0}}
.epirow{{display:flex;flex-direction:column;gap:5px;padding:10px 12px;border-radius:8px;cursor:pointer;
        border:1px solid transparent}}
.epirow:hover{{background:var(--panel-2)}}
.epirow.active{{background:var(--accent-soft);border-color:var(--accent)}}
.epirow .t{{font-size:13px;font-weight:600;text-transform:capitalize;display:flex;justify-content:space-between;gap:8px}}
.epirow .t .rank{{color:var(--ink-3);font-family:var(--mono);font-weight:400;font-size:11px}}
.epirow .meta{{font-size:10.5px;color:var(--ink-3);font-family:var(--mono);display:flex;gap:7px;flex-wrap:wrap}}
.epirow .bars{{display:flex;gap:3px;margin-top:1px}}
.epirow .bar{{height:3px;flex:1;background:var(--sunk);border-radius:2px;overflow:hidden}}
.epirow .bar i{{display:block;height:100%}}
.epirow .bar.sem i{{background:var(--accent)}}
.epirow .bar.sig i{{background:var(--purple)}}
.epirow .bar.suc i{{background:var(--good)}}
.epirow .bar.suc.low i{{background:var(--bad)}}
.epirow .unaud{{color:var(--ink-3);font-style:italic}}
.empty{{padding:40px 18px;text-align:center;color:var(--ink-3);font-size:13px}}

.stage{{display:flex;flex-direction:column;min-width:0;min-height:0;height:100%;overflow-y:auto}}
.stagehead{{padding:18px 26px 12px;border-bottom:1px solid var(--rule);flex:none}}
.stagehead .task{{font-size:20px;font-weight:650;text-transform:capitalize}}
.stagehead .id{{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);margin-top:3px}}
.videowrap{{padding:20px 26px 0;flex:none}}
video{{width:100%;max-height:46vh;border-radius:10px;background:#000;display:block}}
.novideo{{width:100%;aspect-ratio:16/9;border-radius:10px;background:var(--panel-2);display:flex;
         align-items:center;justify-content:center;color:var(--ink-3);font-size:13px}}

.chartwrap{{padding:16px 26px 0;flex:none}}
.chartwrap h3{{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-2);
              margin:0 0 8px;font-weight:600}}
.chartwrap h3 .capnote{{text-transform:none;letter-spacing:0;color:var(--ink-3);font-weight:400;
                        font-family:var(--mono);margin-left:8px}}
.chartlegend{{display:flex;gap:18px;margin-bottom:8px;font-family:var(--mono);font-size:12px}}
.chartlegend .lk{{display:flex;align-items:center;gap:6px;color:var(--ink-2)}}
.chartlegend .dot{{width:8px;height:8px;border-radius:50%;flex:none;display:inline-block}}
.chartlegend b{{color:var(--ink);font-variant-numeric:tabular-nums;margin-left:2px}}
canvas{{width:100%;height:130px;display:block;cursor:pointer;background:var(--sunk);border-radius:8px}}
.dips{{padding:8px 26px 0;font-size:11.5px;color:var(--ink-3);font-family:var(--mono);flex:none}}
.dips b{{color:var(--bad)}}

.scores{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:18px 26px;flex:none}}
.sc{{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:10px 12px}}
.sc .k{{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-3)}}
.sc .v{{font-size:19px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}}
.txt{{padding:0 26px 26px;font-size:13px;color:var(--ink-2);line-height:1.55;font-style:italic;flex:none}}
.placeholder{{flex:1;display:flex;align-items:center;justify-content:center;color:var(--ink-3);font-size:14px}}
</style></head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="sbhead">
      <h1>ego<span>eye</span></h1>
      <form class="searchbox" method="get" action="/">
        <span class="caret">&gt;</span>
        <input name="q" value="{q}" placeholder="search episodes..." autofocus>
      </form>
      <div class="chips">{chips}</div>
    </div>
    <div class="resline">{resline}</div>
    <div class="epilist">{rows}</div>
  </aside>
  <main class="stage" id="stage">{stage}</main>
</div>

<script>
function cssVar(name){{ return getComputedStyle(document.documentElement).getPropertyValue(name); }}

function drawTrace(){{
  if(typeof TRACE === "undefined" || !TRACE) return;
  var canvas=document.getElementById("m-canvas");
  if(!canvas) return;
  var dpr=window.devicePixelRatio||1;
  var w=canvas.clientWidth, h=canvas.clientHeight||130;
  canvas.width=w*dpr; canvas.height=h*dpr;
  var ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);

  var t=TRACE.t, conf=TRACE.conf;
  var tmax=t[t.length-1]||1;
  function x(v){{ return v/tmax*w; }}
  function y(v){{ return h-10-v*(h-20); }}

  ctx.strokeStyle=cssVar("--rule");
  ctx.lineWidth=1;
  [0,0.5,1].forEach(function(v){{ ctx.beginPath(); ctx.moveTo(0,y(v)); ctx.lineTo(w,y(v)); ctx.stroke(); }});

  var accent="#4CC9E8";
  ctx.beginPath(); ctx.moveTo(x(t[0]),h);
  for(var i=0;i<t.length;i++){{ ctx.lineTo(x(t[i]),y(conf[i])); }}
  ctx.lineTo(x(t[t.length-1]),h); ctx.closePath();
  ctx.fillStyle=accent+"26"; ctx.fill();

  ctx.beginPath();
  for(var j=0;j<t.length;j++){{
    if(j===0) ctx.moveTo(x(t[j]),y(conf[j])); else ctx.lineTo(x(t[j]),y(conf[j]));
  }}
  ctx.strokeStyle=accent; ctx.lineWidth=2; ctx.stroke();

  ctx.fillStyle="#F0616B";
  (TRACE.dips||[]).forEach(function(d){{
    ctx.beginPath(); ctx.arc(x(d.t),y(d.conf),4,0,Math.PI*2); ctx.fill();
  }});

  var video=document.getElementById("m-video");
  var ct=video?(video.currentTime||0):0;
  ctx.beginPath(); ctx.moveTo(x(ct),0); ctx.lineTo(x(ct),h);
  ctx.strokeStyle="#E8ECF4"; ctx.lineWidth=1.5; ctx.stroke();
}}

function drawEye(){{
  if(typeof EYE === "undefined" || !EYE) return;
  var canvas=document.getElementById("m-eye-canvas");
  if(!canvas) return;
  var dpr=window.devicePixelRatio||1;
  var w=canvas.clientWidth, h=canvas.clientHeight||130;
  canvas.width=w*dpr; canvas.height=h*dpr;
  var ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);

  // Mirrored around the vertical center: each cycle contributes a top trace
  // (center - v) and a bottom trace (center + v). Cycles start/end near-zero
  // speed (segment_cycles cuts at quiet valleys) and peak mid-cycle, so the
  // mirror pinches shut at the edges and opens in the middle — an actual
  // eye shape, not just an overlaid line chart. Tight overlap = consistent
  // motion = the eye reads "open"; scattered cycles smear the opening shut.
  var n=EYE.median.length;
  var cy=h/2, half=(h-20)/2;
  function x(i){{ return i/(n-1)*w; }}
  function yTop(v){{ return cy-Math.max(0,Math.min(1,v))*half; }}
  function yBot(v){{ return cy+Math.max(0,Math.min(1,v))*half; }}

  ctx.strokeStyle=cssVar("--rule");
  ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(0,cy); ctx.lineTo(w,cy); ctx.stroke();

  function traceBoth(row){{
    ctx.beginPath();
    for(var i=0;i<row.length;i++){{ if(i===0) ctx.moveTo(x(i),yTop(row[i])); else ctx.lineTo(x(i),yTop(row[i])); }}
    ctx.stroke();
    ctx.beginPath();
    for(var j=0;j<row.length;j++){{ if(j===0) ctx.moveTo(x(j),yBot(row[j])); else ctx.lineTo(x(j),yBot(row[j])); }}
    ctx.stroke();
  }}

  // every individual cycle, faint
  ctx.strokeStyle="#4CC9E8"; ctx.globalAlpha=0.18; ctx.lineWidth=1;
  EYE.cycles.forEach(traceBoth);
  ctx.globalAlpha=1;

  // median profile, bold
  ctx.strokeStyle="#4CC9E8"; ctx.lineWidth=2.5;
  traceBoth(EYE.median);
}}

function seekTo(t){{
  var video=document.getElementById("m-video");
  if(video) video.currentTime=t;
  drawTrace();
}}

function onTimeUpdate(){{
  drawTrace();
  var video=document.getElementById("m-video");
  var lvConf=document.getElementById("lv-conf");
  if(!video || typeof TRACE==="undefined" || !TRACE || !lvConf) return;
  var ct=video.currentTime||0;
  var idx=0, best=1e9;
  for(var i=0;i<TRACE.t.length;i++){{
    var d=Math.abs(TRACE.t[i]-ct);
    if(d<best){{ best=d; idx=i; }}
  }}
  lvConf.textContent=TRACE.conf[idx].toFixed(3);
  var lvDip=document.getElementById("lv-dip");
  if(lvDip && TRACE.dips.length){{
    var nearest=TRACE.dips[0], bd=1e9;
    for(var k=0;k<TRACE.dips.length;k++){{
      var dd=Math.abs(TRACE.dips[k].t-ct);
      if(dd<bd){{ bd=dd; nearest=TRACE.dips[k]; }}
    }}
    lvDip.textContent=nearest.t.toFixed(1)+"s";
  }}
}}

(function init(){{
  var video=document.getElementById("m-video");
  if(video) video.ontimeupdate=onTimeUpdate;
  var canvas=document.getElementById("m-canvas");
  if(canvas){{
    canvas.addEventListener("click",function(e){{
      if(typeof TRACE==="undefined" || !TRACE) return;
      var rect=e.target.getBoundingClientRect();
      var frac=(e.clientX-rect.left)/rect.width;
      var tmax=TRACE.t[TRACE.t.length-1]||1;
      seekTo(frac*tmax);
    }});
  }}
  var dipLinks=document.querySelectorAll(".dips .chip");
  for(var i=0;i<dipLinks.length;i++){{
    dipLinks[i].addEventListener("click",function(e){{
      e.preventDefault();
      seekTo(parseFloat(e.target.dataset.t));
    }});
  }}
  drawTrace();
  drawEye();
}})();
window.addEventListener("resize",drawEye);
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="episodes_washdishes.csv")
    ap.add_argument("--results", default="audit_washdishes_local.parquet")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    ep = pd.read_csv(args.episodes, dtype={"episode_id": str})
    STATE["mp4_lookup"] = dict(zip(ep["episode_id"], ep.get("preview_mp4", "")))
    STATE["zarr_lookup"] = dict(zip(ep["episode_id"], ep.get("zarr_path", "")))
    STATE["fps_lookup"] = dict(zip(ep["episode_id"], ep.get("fps", 30.0)))
    STATE["search"] = E.EgoSearch.build(
        args.episodes, results_parquet=args.results,
        annotations_csv=args.annotations, verbose=True,
    )

    print(f"\n  http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
