import { smoothAllFrames } from './smoother'
import type { PoseLandmark } from './pose'

export interface RecordedFrame {
  landmarks: PoseLandmark[]
  timestamp: number
}

export function exportJSON(frames: RecordedFrame[]) {
  if (frames.length < 2) return
  const smoothed = smoothAllFrames(frames.map((f) => f.landmarks), 7)
  const t0 = frames[0].timestamp
  const tN = frames[frames.length - 1].timestamp
  const durationSec = (tN - t0) / 1000
  const fps = frames.length / durationSec

  const payload = {
    version: 1,
    frameCount: frames.length,
    durationSeconds: +durationSec.toFixed(3),
    fps: +fps.toFixed(1),
    frames: frames.map((f, i) => ({
      t: +(f.timestamp - t0).toFixed(1),
      landmarks: smoothed[i].map((lm) => ({
        x: +lm.x.toFixed(4),
        y: +lm.y.toFixed(4),
        z: +lm.z.toFixed(4),
      })),
    })),
  }

  triggerDownload(JSON.stringify(payload, null, 2), 'mocap.json', 'application/json')
}

export function exportHTML(frames: RecordedFrame[]) {
  if (frames.length < 2) return
  const smoothed = smoothAllFrames(frames.map((f) => f.landmarks), 7)
  const t0 = frames[0].timestamp

  // Compact format: [timestamp_ms, [[x,y,z], ...]]
  const frameData = frames.map((f, i) => [
    +(f.timestamp - t0).toFixed(1),
    smoothed[i].map((lm) => [+lm.x.toFixed(4), +lm.y.toFixed(4), +lm.z.toFixed(4)]),
  ])

  triggerDownload(buildPlayerHTML(frameData), 'mocap-player.html', 'text/html')
}

function triggerDownload(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 5000)
}

function buildPlayerHTML(frames: unknown[]): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gym Bro — MoCap Player</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#07111f;color:#f5f7fb;font:14px system-ui;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh;gap:16px;padding:20px}
h1{font-size:20px;letter-spacing:.04em}
canvas{border-radius:16px;max-width:min(640px,100%);background:#07111f;display:block}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:center}
button{padding:10px 22px;border:0;border-radius:999px;background:linear-gradient(135deg,#67f4c7,#8fb8ff);color:#07111f;font:600 14px system-ui;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
.info{color:rgba(245,247,251,.5);font-size:12px;font-variant-numeric:tabular-nums}
</style>
</head>
<body>
<h1>Gym Bro Motion Capture</h1>
<canvas id="c" width="640" height="360"></canvas>
<div class="bar">
  <button id="btn">Play</button>
  <span class="info" id="info">frame 0 / ${frames.length}</span>
</div>
<script>
const FRAMES=${JSON.stringify(frames)};
const L={nose:0,ls:11,rs:12,le:13,re:14,lw:15,rw:16,lh:23,rh:24,lk:25,rk:26,la:27,ra:28};
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
let playing=false,rafId=null,startMs=0;

function pt(lm,i){const p=lm[i];return p?{x:(1-p[0])*cv.width,y:p[1]*cv.height,ok:true}:{x:0,y:0,ok:false}}
function mid(a,b){return{x:(a.x+b.x)/2,y:(a.y+b.y)/2,ok:a.ok&&b.ok}}
function seg(a,b,w,c){if(!a.ok||!b.ok)return;ctx.save();ctx.strokeStyle=c;ctx.lineWidth=w;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.restore()}
function dot(p,r,c){if(!p.ok)return;ctx.save();ctx.fillStyle=c;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();ctx.restore()}

function drawFrame(fi){
  const lm=FRAMES[fi]?.[1]; if(!lm)return;
  const s=cv.width/640;
  ctx.fillStyle='#07111f'; ctx.fillRect(0,0,cv.width,cv.height);
  const{nose,ls,rs,le,re,lw,rw,lh,rh,lk,rk,la,ra}=Object.fromEntries(Object.entries(L).map(([k,i])=>[k,pt(lm,i)]));
  const sm=mid(ls,rs),hm=mid(lh,rh);
  if(ls.ok&&rs.ok&&lh.ok&&rh.ok){ctx.save();ctx.fillStyle='rgba(80,140,255,.18)';ctx.strokeStyle='rgba(80,140,255,.55)';ctx.lineWidth=1.5*s;ctx.beginPath();ctx.moveTo(ls.x,ls.y);ctx.lineTo(rs.x,rs.y);ctx.lineTo(rh.x,rh.y);ctx.lineTo(lh.x,lh.y);ctx.closePath();ctx.fill();ctx.stroke();ctx.restore()}
  seg(sm,hm,6*s,'#b06fff');
  if(nose.ok)seg(sm,nose,8*s,'#5cf9aa');
  for(const[a,b,w]of[[ls,le,13],[le,lw,9],[rs,re,13],[re,rw,9],[lh,lk,13],[lk,la,9],[rh,rk,13],[rk,ra,9]])seg(a,b,w*s,'#5cf9aa');
  for(const p of[ls,rs,le,re,lw,rw,lh,rh,lk,rk,la,ra])dot(p,7*s,'#ffb84d');
  if(nose.ok){ctx.save();ctx.fillStyle='#f5c88a';ctx.strokeStyle='#5cf9aa';ctx.lineWidth=2*s;ctx.beginPath();ctx.arc(nose.x,nose.y,16*s,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.restore()}
  document.getElementById('info').textContent='frame '+(fi+1)+' / '+FRAMES.length;
}

function tick(now){
  if(!playing)return;
  const elapsed=now-startMs;
  let fi=FRAMES.findIndex(f=>f[0]>=elapsed);
  if(fi===-1||fi>=FRAMES.length-1){fi=FRAMES.length-1;playing=false;document.getElementById('btn').textContent='Replay'}
  drawFrame(fi);
  if(playing)rafId=requestAnimationFrame(tick);
}

document.getElementById('btn').addEventListener('click',()=>{
  if(rafId)cancelAnimationFrame(rafId);
  playing=true; startMs=performance.now();
  document.getElementById('btn').textContent='Playing…';
  rafId=requestAnimationFrame(tick);
});

drawFrame(0);
</script>
</body>
</html>`
}
