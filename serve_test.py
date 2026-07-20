"""Local moderation tester — serves an upload page and runs the REAL worker pipeline:
  • IMAGE → the HYBRID gate (gate.decide_image): NudeNet body-parts (free) → Claude Haiku vision.
  • VIDEO → NudeNet only across sampled frames (no Claude, matching the worker).
Shows the per-body-part breakdown + how it was decided. No DB/R2/schema needed.

Run:   ./.venv/bin/python serve_test.py   (or `npm run dev`)   Then open http://localhost:9000
"""
import io
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Load ANTHROPIC_API_KEY (for the vision layer) from env, else a repo-local .env.local (gitignored)
# — BEFORE importing config, which snapshots it. Without a key the tester still runs (NudeNet-only).
if not os.environ.get("ANTHROPIC_API_KEY"):
    _envf = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")
    try:
        for _line in open(_envf):
            if _line.strip().startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = _line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    except OSError:
        pass

from PIL import Image  # noqa: E402

import config  # noqa: E402
import frames  # noqa: E402
import gate  # noqa: E402
import nsfw  # noqa: E402

PORT = int(os.environ.get("PORT", "9000"))
MAX_BODY = 500 * 1024 * 1024

# The labels that actually drive the gate (union of block + flag) — always shown in the tester so
# there's a full % board even when the image is clean, each with its block/flag threshold.
GATING_LABELS = sorted(set(config.BLOCK_THRESHOLDS) | set(config.FLAG_THRESHOLDS))
THRESHOLDS = {l: {"block": config.BLOCK_THRESHOLDS.get(l), "flag": config.FLAG_THRESHOLDS.get(l)}
              for l in GATING_LABELS}
MIN_PROB = 0.2  # NudeNet v3's hardcoded postprocess floor (detections below this never surface)


def _meta():
    # all_labels = the full v3 class list so the tester shows the COMPLETE breakdown (every body
    # part), not just the gating ones.
    return {"thresholds": THRESHOLDS, "gating": GATING_LABELS, "min_prob": MIN_PROB,
            "all_labels": nsfw.LABELS}


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VRE NSFW Tester</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;font:15px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;background:#0d0d0d;color:#eee}
.wrap{max-width:680px;margin:0 auto;padding:28px 20px 70px}
h1{font-size:20px;margin:0 0 4px}.sub{color:#888;font-size:13px;margin:0 0 22px}
#drop{border:2px dashed #333;border-radius:16px;padding:32px 20px;text-align:center;cursor:pointer;transition:.15s;background:#141414}
#drop.hover{border-color:#14F195;background:#14f1950d}#drop small{color:#777;display:block;margin-top:6px}
#preview{margin:16px 0;text-align:center}#preview img,#preview video{max-width:100%;max-height:300px;border-radius:12px;background:#000}
.card{margin-top:16px;padding:20px;border-radius:16px;background:#161616;border:1px solid #262626;display:none}
.badge{display:inline-block;padding:7px 16px;border-radius:999px;font-weight:800;font-size:15px}
.approve{background:#0f2e1f;color:#14F195}.flag{background:#3a3212;color:#EAB308}.reject{background:#3a1414;color:#ff5a5a}
.reason{color:#aaa;font-size:13px;margin:8px 0 4px}
.rows{margin-top:14px}
.r{display:flex;align-items:center;gap:10px;margin:7px 0}
.r .lab{width:190px;font-size:12.5px;font-family:ui-monospace,Menlo,monospace;color:#cfcfcf;flex-shrink:0}
.r .lab.hot{color:#fff;font-weight:700}
.r .bar{position:relative;flex:1;height:12px;border-radius:999px;background:#242424;overflow:hidden}
.r .fill{height:100%;border-radius:999px}
.r .mark{position:absolute;top:0;bottom:0;width:2px}
.r .mark.blk{background:#ff5a5a}.r .mark.flg{background:#EAB308}
.r .pc{width:48px;text-align:right;font-variant-numeric:tabular-nums;font-size:13px;flex-shrink:0}
.r .thr{width:94px;text-align:right;font-size:11px;color:#888;font-family:ui-monospace,Menlo,monospace;flex-shrink:0}
.clean{color:#14F195;margin-top:12px}
.err{color:#ff7a7a}
.note{margin-top:26px;color:#777;font-size:12px;border-top:1px solid #222;padding-top:14px}
.spin{display:inline-block;width:16px;height:16px;border:2px solid #14F195;border-top-color:transparent;border-radius:50%;animation:s .8s linear infinite;vertical-align:-3px;margin-right:8px}
@keyframes s{to{transform:rotate(360deg)}}
</style></head><body><div class="wrap">
<h1>VRE Moderation Tester — hybrid gate</h1>
<p class="sub">IMAGE → NudeNet (free) then Claude Haiku vision · VIDEO → NudeNet only (no Claude) · shows the body-part breakdown + how it was decided</p>
<div id="drop">📤 <b>Drop an image or video</b>, or click to choose<small>image = full hybrid (a Claude call, ~2s) · video = frame-sampled NudeNet (a few seconds)</small><input id="file" type="file" accept="image/*,video/*" hidden></div>
<div id="preview"></div>
<div class="card" id="card"></div>
<div class="note">Only the <b>EXPOSED_*</b> parts drive the gate. A row turns <b>white/bold</b> when it crosses a threshold (the reason for a flag/reject). Tune thresholds in <code>config.py</code> / env, then reload.</div>
</div><script>
const drop=document.getElementById('drop'),file=document.getElementById('file'),prev=document.getElementById('preview'),card=document.getElementById('card');
drop.onclick=()=>file.click();
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hover')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hover')}));
drop.addEventListener('drop',ev=>{if(ev.dataTransfer.files[0])go(ev.dataTransfer.files[0])});
file.onchange=()=>{if(file.files[0])go(file.files[0])};
const pct=x=>(x*100).toFixed(1)+'%';
const clr=x=> x>=0.85?'#ff5a5a' : x>=0.5?'#EAB308' : '#14F195';
async function go(f){
  const isVideo=f.type.startsWith('video')||/\.(mp4|mov|webm|mkv|avi|m4v)$/i.test(f.name);
  prev.innerHTML=isVideo?`<video src="${URL.createObjectURL(f)}" controls muted></video>`:`<img src="${URL.createObjectURL(f)}">`;
  card.style.display='block';card.innerHTML=`<span class="spin"></span>Scanning ${isVideo?'video frames':'image'}…`;
  try{
    const r=await fetch('/classify',{method:'POST',headers:{'X-Kind':isVideo?'video':'image'},body:f});
    const d=await r.json();
    if(d.error){card.innerHTML=`<span class="err">Error: ${d.error}</span>`;return}
    const cls=d.status==='rejected'?'reject':d.status==='flagged'?'flag':'approve';
    const label={rejected:'REJECT',flagged:'FLAG',approved:'APPROVE'}[d.status];
    const sc=d.scores||{}, th=d.thresholds||{}, INF=99;
    // show the COMPLETE body-part list (all v3 labels, even at 0%), plus anything else detected
    const labs=[...(d.all_labels||d.gating||[])];
    Object.keys(sc).forEach(k=>{if(!labs.includes(k))labs.push(k)});
    const rowsArr=labs.map(k=>({k,v:sc[k]||0,t:th[k]||{}})).sort((a,b)=>b.v-a.v||a.k.localeCompare(b.k));
    const rows=rowsArr.map(({k,v,t})=>{
      const blk=t.block==null?INF:t.block, flg=t.flag==null?INF:t.flag;
      const col = v>=blk?'#ff5a5a' : v>=flg?'#EAB308' : (v>0?'#14F195':'#333');
      const hot = k===d.reason_label;
      const bm = t.block!=null?`<div class="mark blk" style="left:${Math.min(100,t.block*100)}%"></div>`:'';
      const fm = t.flag!=null?`<div class="mark flg" style="left:${Math.min(100,t.flag*100)}%"></div>`:'';
      const thTxt = (t.block!=null||t.flag!=null)?`blk ${t.block!=null?Math.round(t.block*100):'–'} · flg ${t.flag!=null?Math.round(t.flag*100):'–'}`:'';
      return `<div class="r" style="opacity:${v===0?.5:1}"><div class="lab ${hot?'hot':''}">${k}</div>
        <div class="bar"><div class="fill" style="width:${Math.min(100,v*100)}%;background:${col}"></div>${bm}${fm}</div>
        <div class="pc">${pct(v)}</div><div class="thr">${thTxt}</div></div>`;
    }).join('');
    let decision='';
    if(d.kind==='video') decision=`<div class="reason">🎞 <b>video</b> · NudeNet only (no Claude) · ${d.frames} frames sampled (max per label)</div>`;
    else if(d.layer==='nudenet') decision=`<div class="reason">🛡 <b>NudeNet</b> free block · ${d.reason_label||''} ${d.reason_label?pct(d.reason_score):''}</div>`;
    else if(d.layer==='vision') decision=`<div class="reason">🧠 <b>Claude vision</b> · ${d.decision||''}</div>`;
    else if(d.layer==='nudenet-only') decision=`<div class="reason">🛡 <b>NudeNet only</b> (vision off) · ${d.decision||''}</div>`;
    else if(d.layer==='fallback') decision=`<div class="reason">⚠ <b>vision error → held</b> · ${d.decision||''}</div>`;
    const nnReason = d.reason_label ? `<div class="reason" style="opacity:.7">NudeNet flag: <b>${d.reason_label}</b> ${pct(d.reason_score)}</div>` : '';
    const floor = `<div class="reason">surfacing detections ≥ ${pct(d.min_prob)} · <span style="color:#ff5a5a">▏</span>block thr <span style="color:#EAB308">▏</span>flag thr</div>`;
    card.innerHTML=`<div><span class="badge ${cls}">${label}</span></div>${decision}${nnReason}${floor}<div class="rows">${rows}</div>`;
  }catch(e){card.innerHTML=`<span class="err">Request failed: ${e}</span>`}
}
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path.split("?")[0] != "/classify":
            self._send(404, b'{"error":"not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                self._send(413, b'{"error":"empty or too large"}')
                return
            body = self.rfile.read(length)
            if self.headers.get("X-Kind") == "video":
                with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
                    tf.write(body)
                    tf.flush()
                    fr = frames.sample(tf.name)
                if not fr:
                    self._send(200, json.dumps({"kind": "video", "error": "no frames sampled (is ffmpeg installed?)"}).encode())
                    return
                scores = nsfw.detect_max(fr)
                status, rlabel, rscore = config.verdict(scores)
                self._send(200, json.dumps({"kind": "video", "scores": scores, "status": status,
                                            "reason_label": rlabel, "reason_score": rscore, "frames": len(fr),
                                            **_meta()}).encode())
            else:
                im = Image.open(io.BytesIO(body)).convert("RGB")
                d = gate.decide_image(im)                    # full hybrid: NudeNet → Claude vision
                scores = d.get("scores") or {}
                _, rlabel, rscore = config.verdict(scores)   # NudeNet detail for the bars
                self._send(200, json.dumps({"kind": "image", "scores": scores, "status": d["status"],
                                            "reason_label": rlabel, "reason_score": rscore,
                                            "layer": d.get("layer"), "decision": d.get("reason", ""),
                                            **_meta()}).encode())
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"NSFW tester → http://localhost:{PORT}   (NudeNet v3)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
