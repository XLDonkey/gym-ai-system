"""
XL Fitness AI — Skeleton Viewer (CPU mode)
Single-person pose detection using YOLO on CPU.
Viewer: http://<PI_IP>:8080
WebSocket: ws://<PI_IP>:8765
"""
import asyncio, json, threading, time, sys
import websockets
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np

HTML = """<!DOCTYPE html>
<html><head>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no"/>
<title>XL Fitness AI</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#06040f;overflow:hidden;}
canvas{width:100vw;height:100vh;display:block;}
#hdr{position:fixed;top:0;left:0;right:0;padding:10px 16px;background:rgba(6,4,15,.85);display:flex;align-items:center;justify-content:space-between;z-index:99;}
.logo{font-family:sans-serif;font-weight:900;letter-spacing:4px;color:#a855f7;font-size:.9rem;}
#fps{position:fixed;bottom:10px;right:10px;color:#555;font-family:monospace;font-size:.7rem;}
#msg{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);color:#444;font-family:sans-serif;font-size:1.1rem;}
</style>
</head><body>
<div id="hdr"><span class="logo">XLFITNESS AI</span></div>
<canvas id="c"></canvas>
<div id="fps">--</div>
<div id="msg">Stand in front of camera</div>
<script>
const c=document.getElementById('c'),ctx=c.getContext('2d');
const fps=document.getElementById('fps'),msg=document.getElementById('msg');
function resize(){c.width=window.innerWidth;c.height=window.innerHeight;}
resize();window.addEventListener('resize',resize);
const SKEL=[[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
let fc=0,ft=Date.now();
const ws=new WebSocket('ws://'+location.hostname+':8765');
ws.onmessage=(e)=>{
  const d=JSON.parse(e.data);
  const kps=d.kps||[];
  ctx.clearRect(0,0,c.width,c.height);
  if(kps.length>=17){
    msg.style.display='none';
    const W=c.width,H=c.height;
    ctx.strokeStyle='#a855f7';ctx.lineWidth=3;ctx.lineCap='round';
    for(const [a,b] of SKEL){
      if(kps[a]&&kps[b]&&kps[a].c>0.3&&kps[b].c>0.3){
        ctx.beginPath();ctx.moveTo(kps[a].x*W,kps[a].y*H);ctx.lineTo(kps[b].x*W,kps[b].y*H);ctx.stroke();
      }
    }
    for(const kp of kps){
      if(kp&&kp.c>0.3){
        ctx.beginPath();ctx.arc(kp.x*W,kp.y*H,5,0,Math.PI*2);ctx.fillStyle='#e879f9';ctx.fill();
      }
    }
  } else {
    msg.style.display='block';
  }
  fc++;const now=Date.now();if(now-ft>=1000){fps.textContent=fc+'fps';fc=0;ft=now;}
};
ws.onclose=()=>{msg.textContent='Reconnecting...';msg.style.display='block';setTimeout(()=>location.reload(),2000);};
</script></body></html>"""

# ── Global state ──────────────────────────────────────────────────────────────
latest_kps = []
kps_lock = threading.Lock()

# ── Step 1: Load YOLO model ───────────────────────────────────────────────────
print("Step 1: Loading YOLO pose model...", flush=True)
model = YOLO('/home/xlraspberry2026/yolo11n-pose.pt')
print("YOLO loaded OK.", flush=True)

# ── Step 2: Start camera ──────────────────────────────────────────────────────
print("Step 2: Starting camera...", flush=True)
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={'size': (640, 480), 'format': 'RGB888'}
)
picam2.configure(config)
picam2.start()
time.sleep(2)
print("Camera started OK.", flush=True)

# ── Step 3: Pose detection loop (background thread) ───────────────────────────
def pose_loop():
    global latest_kps
    print("Pose loop running...", flush=True)
    while True:
        try:
            frame = picam2.capture_array()
            results = model(frame, imgsz=320, verbose=False, conf=0.4)
            kps_list = []
            if results and len(results) > 0:
                r = results[0]
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    kp_data = r.keypoints.data[0].cpu().numpy()
                    h, w = frame.shape[:2]
                    for kp in kp_data:
                        kps_list.append({
                            'x': round(float(kp[0]) / w, 4),
                            'y': round(float(kp[1]) / h, 4),
                            'c': round(float(kp[2]), 3)
                        })
            with kps_lock:
                latest_kps = kps_list
        except Exception as ex:
            print(f"Pose error: {ex}", flush=True)
            time.sleep(0.5)

t = threading.Thread(target=pose_loop, daemon=True)
t.start()

# ── Step 4: WebSocket + HTTP servers ─────────────────────────────────────────
async def ws_handler(ws):
    try:
        while True:
            with kps_lock:
                kps = latest_kps[:]
            await ws.send(json.dumps({'kps': kps}))
            await asyncio.sleep(0.1)
    except:
        pass

async def http_handler(reader, writer):
    await reader.readline()
    while True:
        line = await reader.readline()
        if line in (b'\r\n', b''):
            break
    html = HTML.encode()
    writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: '
                 + str(len(html)).encode() + b'\r\n\r\n' + html)
    await writer.drain()
    writer.close()

async def main():
    ws_server = await websockets.serve(ws_handler, '0.0.0.0', 8765)
    http_server = await asyncio.start_server(http_handler, '0.0.0.0', 8080)
    print('Skeleton viewer: http://192.168.1.40:8080', flush=True)
    print('WebSocket: ws://192.168.1.40:8765', flush=True)
    await asyncio.gather(ws_server.serve_forever(), http_server.serve_forever())

print("Step 4: Starting servers...", flush=True)
asyncio.run(main())
