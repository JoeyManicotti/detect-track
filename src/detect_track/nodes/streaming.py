"""
HTTP streaming server — serves annotated video frames as MJPEG and
track metadata as JSON so any browser can consume the pipeline output.

Endpoints
─────────
GET /          Minimal self-contained HTML viewer (no CDN dependencies)
GET /video     MJPEG stream  (multipart/x-mixed-replace)
GET /tracks    JSON snapshot of the latest track state
GET /health    Plain-text health check

Design
──────
The server runs in a daemon thread inside the main process.  The main
loop calls ``server.push_frame(frame_bgr, result)`` after annotating
each frame; the MJPEG handler loop picks up the latest JPEG and sends
it to every connected client.

Thread safety
─────────────
A single ``threading.Lock`` guards ``_current_jpeg`` and
``_current_tracks``.  Clients that connect while no frame is available
busy-wait with a short sleep.  ``ThreadingMixIn`` gives each HTTP
connection its own thread so one slow client cannot block others.
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import socketserver
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Optional

import cv2
import numpy as np

from detect_track.ipc.messages import TrackResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Self-contained browser UI (no CDN, fully offline)
# ---------------------------------------------------------------------------

_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>detect-track</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0d;color:#e2e2e2;font-family:'Menlo','Consolas','Monaco',monospace;
     display:flex;height:100vh;overflow:hidden}
/* ── upload panel ── */
#upload-panel{flex:1;display:flex;flex-direction:column;align-items:center;
              justify-content:center;gap:18px;padding:40px}
#upload-panel.hidden{display:none}
#drop-zone{border:2px dashed #2a2a2a;border-radius:8px;padding:48px 64px;
           text-align:center;cursor:pointer;transition:border-color .2s}
#drop-zone:hover,#drop-zone.drag{border-color:#0f0}
#drop-zone input{display:none}
#drop-zone label{cursor:pointer;display:block}
#dz-icon{font-size:36px;margin-bottom:12px;color:#333}
#dz-text{font-size:13px;color:#666;margin-bottom:4px}
#dz-sub{font-size:11px;color:#333}
#selected-file{font-size:11px;color:#aaa;min-height:16px;text-align:center}
#upload-btn{background:#0f0;color:#000;border:none;padding:9px 28px;
            border-radius:4px;font-family:inherit;font-size:12px;
            font-weight:bold;cursor:pointer;letter-spacing:1px;
            transition:opacity .15s}
#upload-btn:disabled{opacity:.3;cursor:default}
#upload-progress{width:320px;height:3px;background:#1a1a1a;border-radius:2px;
                 display:none;overflow:hidden}
#upload-bar{height:100%;width:0;background:#0f0;transition:width .1s}
#upload-msg{font-size:11px;color:#555;text-align:center;min-height:16px}
/* ── video panel ── */
#video-panel{flex:1;display:flex;align-items:center;justify-content:center;
             background:#000;overflow:hidden;position:relative}
#video-panel.hidden{display:none}
#stream{max-width:100%;max-height:100%;display:block;object-fit:contain}
#stream-status{position:absolute;top:10px;left:10px;font-size:11px;
               background:rgba(0,0,0,.65);padding:3px 9px;border-radius:4px;color:#0f0}
#stream-status.warn{color:#fa0}
#new-video-btn{position:absolute;top:10px;right:10px;font-family:inherit;
               font-size:11px;background:rgba(0,0,0,.65);color:#aaa;
               border:1px solid #333;padding:3px 10px;border-radius:4px;
               cursor:pointer;letter-spacing:.5px}
#new-video-btn:hover{color:#fff;border-color:#555}
/* ── sidebar ── */
#sidebar{width:270px;min-width:270px;background:#141414;border-left:1px solid #2a2a2a;
         display:flex;flex-direction:column;overflow:hidden}
#sidebar-header{padding:14px 16px;background:#1a1a1a;border-bottom:1px solid #2a2a2a;
                font-size:11px;letter-spacing:2px;color:#0f0;text-transform:uppercase;flex-shrink:0}
#stats-bar{display:flex;gap:12px;padding:8px 16px;font-size:11px;color:#555;
           border-bottom:1px solid #1f1f1f;flex-shrink:0}
.sv{color:#ccc}
#tracks-el{flex:1;overflow-y:auto;padding:10px}
#tracks-el::-webkit-scrollbar{width:4px}
#tracks-el::-webkit-scrollbar-track{background:#111}
#tracks-el::-webkit-scrollbar-thumb{background:#333;border-radius:2px}
.card{background:#1c1c1c;border:1px solid #2a2a2a;border-left-width:3px;border-radius:4px;
      margin-bottom:8px;padding:10px 12px}
.row{display:flex;justify-content:space-between;align-items:center}
.tid{font-size:12px;font-weight:bold}
.score{font-size:11px;color:#888}
.label{font-size:11px;color:#aaa;margin-top:3px}
.box{font-size:10px;color:#444;margin-top:4px}
.empty{color:#333;text-align:center;padding:40px 0;font-size:11px;line-height:1.9}
</style>
</head>
<body>

<!-- Upload panel -->
<div id="upload-panel">
  <div id="drop-zone">
    <input type="file" id="file-input" accept="video/*">
    <label for="file-input">
      <div id="dz-icon">&#128249;</div>
      <div id="dz-text">Click to select a video file</div>
      <div id="dz-sub">or drag and drop here</div>
    </label>
  </div>
  <div id="selected-file"></div>
  <button id="upload-btn" disabled>Upload &amp; Analyze</button>
  <div id="upload-progress"><div id="upload-bar"></div></div>
  <div id="upload-msg"></div>
</div>

<!-- Video panel (hidden until processing starts) -->
<div id="video-panel" class="hidden">
  <img id="stream" src="" alt="video stream">
  <div id="stream-status">\u25cf LIVE</div>
  <button id="new-video-btn">+ New Video</button>
</div>

<!-- Sidebar always visible -->
<div id="sidebar">
  <div id="sidebar-header">detect-track</div>
  <div id="stats-bar">
    <span>frame <span class="sv" id="sf">\u2013</span></span>
    <span>tracks <span class="sv" id="sc">0</span></span>
  </div>
  <div id="tracks-el"><div class="empty">Upload a video to begin.</div></div>
</div>

<script>
// ── colour palette ───────────────────────────────────────────────
const cc={};
function hsv2rgb(h,s,v){const i=Math.floor(h*6),f=h*6-i,p=v*(1-s),q=v*(1-f*s),t=v*(1-(1-f)*s);
  return[[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i%6].map(x=>~~(x*255))}
function col(id){if(!cc[id]){const[r,g,b]=hsv2rgb((id*0.6180339887)%1,.85,.95);
  cc[id]=`rgb(${r},${g},${b})`}return cc[id]}

// ── DOM refs ─────────────────────────────────────────────────────
const uploadPanel=document.getElementById('upload-panel'),
      videoPanel=document.getElementById('video-panel'),
      fileInput=document.getElementById('file-input'),
      selFile=document.getElementById('selected-file'),
      uploadBtn=document.getElementById('upload-btn'),
      uploadProgress=document.getElementById('upload-progress'),
      uploadBar=document.getElementById('upload-bar'),
      uploadMsg=document.getElementById('upload-msg'),
      dropZone=document.getElementById('drop-zone'),
      stream=document.getElementById('stream'),
      streamStatus=document.getElementById('stream-status'),
      newVideoBtn=document.getElementById('new-video-btn'),
      sf=document.getElementById('sf'),
      sc=document.getElementById('sc'),
      te=document.getElementById('tracks-el');

// ── file selection ────────────────────────────────────────────────
let selectedFile=null;
function setFile(f){
  selectedFile=f;
  selFile.textContent=f?f.name+' ('+Math.round(f.size/1024/1024)+'MB)':'';
  uploadBtn.disabled=!f;
  uploadMsg.textContent='';
}
fileInput.onchange=()=>setFile(fileInput.files[0]||null);

// Drag-and-drop
dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('drag')});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('drag'));
dropZone.addEventListener('drop',e=>{
  e.preventDefault();dropZone.classList.remove('drag');
  const f=e.dataTransfer.files[0];
  if(f&&f.type.startsWith('video/')){setFile(f);}
});

// ── upload ────────────────────────────────────────────────────────
// Read a cookie by name — needed for JupyterHub XSRF token.
function getCookie(name){
  for(const c of document.cookie.split(';')){
    const[k,v]=c.trim().split('=',2);
    if(k===name)return decodeURIComponent(v||'');
  }
  return '';
}

uploadBtn.onclick=async()=>{
  if(!selectedFile)return;
  uploadBtn.disabled=true;
  uploadProgress.style.display='block';
  uploadBar.style.width='0%';
  uploadMsg.textContent='Uploading\u2026';

  const fd=new FormData();
  fd.append('file',selectedFile);
  const xhr=new XMLHttpRequest();
  xhr.open('POST','/upload');
  // JupyterHub (and similar proxies) require X-XSRFToken on POST requests.
  const xsrf=getCookie('_xsrf');
  if(xsrf) xhr.setRequestHeader('X-XSRFToken',xsrf);
  xhr.upload.onprogress=e=>{
    if(e.lengthComputable)uploadBar.style.width=(e.loaded/e.total*100)+'%';
  };
  xhr.onload=()=>{
    if(xhr.status===202){
      uploadMsg.textContent='Processing\u2026';
      uploadBar.style.width='100%';
      showVideoPanel();
    } else {
      uploadMsg.textContent='Upload failed ('+xhr.status+'). Check server logs.';
      uploadBtn.disabled=false;
    }
  };
  xhr.onerror=()=>{uploadMsg.textContent='Network error.';uploadBtn.disabled=false;};
  xhr.send(fd);
};

// ── panel transitions ─────────────────────────────────────────────
function showUploadPanel(){
  uploadPanel.classList.remove('hidden');
  videoPanel.classList.add('hidden');
  stream.src='';
  setFile(null);
  uploadBtn.disabled=true;
  uploadProgress.style.display='none';
  uploadMsg.textContent='';
  te.innerHTML='<div class="empty">Upload a video to begin.</div>';
  sf.textContent='\u2013'; sc.textContent='0';
}
function showVideoPanel(){
  uploadPanel.classList.add('hidden');
  videoPanel.classList.remove('hidden');
  stream.src='/video?'+Date.now();
  streamStatus.textContent='\u25cf LIVE';
  streamStatus.className='';
}

newVideoBtn.onclick=()=>{
  const xsrf=getCookie('_xsrf');
  fetch('/stop',{method:'POST',headers:xsrf?{'X-XSRFToken':xsrf}:{}}).catch(()=>{});
  showUploadPanel();
};

// ── MJPEG reconnect ───────────────────────────────────────────────
stream.onerror=()=>{
  streamStatus.textContent='\u25cb reconnecting\u2026';
  streamStatus.className='warn';
  setTimeout(()=>{if(!videoPanel.classList.contains('hidden')){
    stream.src='/video?'+Date.now();
    streamStatus.textContent='\u25cf LIVE';
    streamStatus.className='';
  }},2000);
};

// ── state polling ─────────────────────────────────────────────────
let lastState='';
async function pollState(){
  try{
    const d=await(await fetch('/state',{cache:'no-store'})).json();
    if(d.state!==lastState){
      lastState=d.state;
      if(d.state==='idle'||d.state==='done'||d.state==='error'){
        showUploadPanel();
        if(d.state==='done') uploadMsg.textContent='Processing complete. Upload another video.';
        if(d.state==='error') uploadMsg.textContent='An error occurred. Try again.';
      } else if(d.state==='loading'||d.state==='running'){
        if(videoPanel.classList.contains('hidden')) showVideoPanel();
      }
    }
  }catch(_){}
  setTimeout(pollState,500);
}
pollState();

// ── track polling ─────────────────────────────────────────────────
async function pollTracks(){
  try{
    const d=await(await fetch('/tracks',{cache:'no-store'})).json();
    sf.textContent=d.frame_id; sc.textContent=d.tracks.length;
    if(!d.tracks.length){
      const msg=lastState==='running'?'No active tracks':'Upload a video to begin.';
      te.innerHTML='<div class="empty">'+msg+'</div>';
    } else {
      te.innerHTML=d.tracks.map(t=>{
        const c=col(t.id),b=t.box?`[${t.box.map(v=>~~v).join(', ')}]`:'\u2013';
        return`<div class="card" style="border-left-color:${c}">
          <div class="row"><span class="tid" style="color:${c}">#${t.id}</span>
          <span class="score">${(t.score*100).toFixed(1)}%</span></div>
          <div class="label">${t.label}</div><div class="box">${b}</div></div>`}).join('');
    }
  }catch(_){}
  setTimeout(pollTracks,250);
}
pollTracks();
</script>
</body>
</html>
""".encode("utf-8")


# ---------------------------------------------------------------------------
# Threaded HTTP server
# ---------------------------------------------------------------------------

class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Each client connection runs in its own thread."""
    daemon_threads = True
    allow_reuse_address = True


class StreamingServer:
    """
    Lightweight MJPEG / JSON streaming server with video-upload support.

    Two usage modes
    ───────────────
    **Inline** (original ``run`` command): caller manages the pipeline loop
    and calls ``push_frame()`` directly::

        server = StreamingServer(host="0.0.0.0", port=8080)
        server.start()
        for result in pipeline.results():
            server.push_frame(annotate(frame, result), result)
        server.stop()

    **Server** (new ``serve`` command): server accepts video uploads and
    signals the caller when a new file is ready::

        server = StreamingServer(host="0.0.0.0", port=8080)
        server.start()
        while True:
            video_path = server.wait_for_upload()   # blocks
            if video_path is None:
                break
            server.set_pipeline_state("loading")
            with Pipeline(config_for(video_path)) as pipeline:
                server.set_pipeline_state("running")
                while not server.stop_requested():
                    result = pipeline.get_result(0.033)
                    ...
                    server.push_frame(frame, result)
            server.set_pipeline_state("done")
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = port

        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._frame_id: int = 0
        self._tracks: List[dict] = []
        self._push_count: int = 0
        self._push_times: List[float] = []
        self._start_time: float = time.monotonic()

        # Server-mode state
        self._pipeline_state: str = "idle"   # idle|loading|running|done|error
        self._upload_queue: queue.Queue = queue.Queue()
        self._stop_event: threading.Event = threading.Event()
        self._upload_dir: Optional[str] = None

        self._server: Optional[_ThreadedHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API — stream push
    # ------------------------------------------------------------------

    def push_frame(
        self,
        frame_bgr: np.ndarray,
        result: Optional[TrackResult] = None,
        *,
        jpeg_quality: int = 80,
    ) -> None:
        """Encode *frame_bgr* as JPEG and broadcast to connected MJPEG clients."""
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if not ok:
            return

        tracks: List[dict] = []
        if result is not None:
            tracks = [
                {
                    "id": t.track_id,
                    "label": t.label,
                    "score": round(float(t.score), 3),
                    "box": [round(v, 1) for v in t.box] if t.box else None,
                }
                for t in result.tracks
            ]

        now = time.monotonic()
        with self._lock:
            self._jpeg = buf.tobytes()
            self._frame_id += 1
            self._tracks = tracks
            self._push_count += 1
            self._push_times.append(now)
            cutoff = now - 2.0
            self._push_times = [t for t in self._push_times if t > cutoff]

        if self._push_count == 1:
            logger.info("First frame pushed to streaming server.")
        elif self._push_count % 100 == 0:
            with self._lock:
                n = len(self._push_times)
                span = (self._push_times[-1] - self._push_times[0]) if n > 1 else 1
                fps = (n - 1) / span if span > 0 else 0
            logger.info(
                "Stream: %d frames pushed total, %.1f fps (last 2s)",
                self._push_count, fps,
            )

    # ------------------------------------------------------------------
    # Public API — server mode
    # ------------------------------------------------------------------

    def set_pipeline_state(self, state: str) -> None:
        """Update the pipeline state reported by GET /state."""
        with self._lock:
            self._pipeline_state = state
        logger.info("Pipeline state → %s", state)

    def wait_for_upload(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Block until a video file is uploaded via POST /upload.

        Returns the path to the saved temp file, or *None* if the server
        is stopping.  The caller is responsible for deleting the file.
        """
        self._stop_event.clear()
        try:
            return self._upload_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop_requested(self) -> bool:
        """Return True if POST /stop was received (user clicked 'New Video')."""
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        self._upload_dir = tempfile.mkdtemp(prefix="dt_uploads_")
        streaming = self  # closure capture

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
                pass

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?")[0]
                if path == "/video":
                    self._mjpeg()
                elif path == "/tracks":
                    self._json_tracks()
                elif path == "/status":
                    self._json_status()
                elif path == "/state":
                    self._json_state()
                elif path in ("/", "/index.html"):
                    self._html()
                elif path == "/health":
                    self._text(b"ok")
                else:
                    self.send_error(404)

            def do_OPTIONS(self) -> None:  # noqa: N802
                """Handle CORS preflight — some proxies send this before POST."""
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 "Content-Type, X-XSRFToken")
                self.send_header("Access-Control-Max-Age", "86400")
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                path = self.path.split("?")[0]
                if path == "/upload":
                    self._handle_upload()
                elif path == "/stop":
                    streaming._stop_event.set()
                    streaming.set_pipeline_state("idle")
                    self._json({"ok": True})
                else:
                    self.send_error(404)

            # ── MJPEG stream ───────────────────────────────────────
            def _mjpeg(self) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                last_id = -1
                while True:
                    try:
                        with streaming._lock:
                            fid = streaming._frame_id
                            jpeg = streaming._jpeg

                        if jpeg is None or fid == last_id:
                            time.sleep(0.01)
                            continue

                        last_id = fid
                        self.wfile.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            + jpeg
                            + b"\r\n"
                        )
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break

            # ── File upload ────────────────────────────────────────
            def _handle_upload(self) -> None:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self.send_error(400, "Expected multipart/form-data")
                    return

                # Parse boundary
                boundary = None
                for part in content_type.split(";"):
                    p = part.strip()
                    if p.lower().startswith("boundary="):
                        boundary = p[9:].strip('"').encode()
                        break
                if not boundary:
                    self.send_error(400, "Missing boundary")
                    return

                length = int(self.headers.get("Content-Length", 0))
                if length <= 0:
                    self.send_error(400, "Empty body")
                    return

                # Read and extract file bytes from multipart body
                data = self.rfile.read(length)
                file_bytes = _parse_multipart_file(data, boundary)
                if file_bytes is None:
                    self.send_error(400, "Could not parse file from upload")
                    return

                # Save to temp file
                suffix = ".mp4"
                tmp = os.path.join(streaming._upload_dir or "/tmp", f"upload_{int(time.time())}{suffix}")
                with open(tmp, "wb") as f:
                    f.write(file_bytes)
                logger.info("Uploaded video saved: %s (%d bytes)", tmp, len(file_bytes))

                # Signal main loop
                streaming.set_pipeline_state("loading")
                streaming._upload_queue.put(tmp)
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"ok":1}')

            # ── State JSON ─────────────────────────────────────────
            def _json_state(self) -> None:
                with streaming._lock:
                    state = streaming._pipeline_state
                body = json.dumps({"state": state}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            # ── Status JSON ────────────────────────────────────────
            def _json_status(self) -> None:
                with streaming._lock:
                    n = len(streaming._push_times)
                    span = (streaming._push_times[-1] - streaming._push_times[0]) if n > 1 else 1
                    fps = round((n - 1) / span, 1) if span > 0 and n > 1 else 0.0
                    body = json.dumps({
                        "uptime_s": round(time.monotonic() - streaming._start_time, 1),
                        "frames_pushed": streaming._push_count,
                        "stream_fps": fps,
                        "latest_frame_id": streaming._frame_id,
                        "active_tracks": len(streaming._tracks),
                        "pipeline_state": streaming._pipeline_state,
                    }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            # ── JSON track snapshot ────────────────────────────────
            def _json_tracks(self) -> None:
                with streaming._lock:
                    body = json.dumps(
                        {"frame_id": streaming._frame_id, "tracks": streaming._tracks}
                    ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            # ── HTML viewer ────────────────────────────────────────
            def _html(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_INDEX_HTML)))
                self.end_headers()
                self.wfile.write(_INDEX_HTML)

            # ── Helpers ────────────────────────────────────────────
            def _text(self, body: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj: dict) -> None:
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

        self._server = _ThreadedHTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="StreamingServer"
        )
        self._thread.start()
        logger.info(
            "Streaming server started → http://%s:%d", self.host, self.port
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._upload_dir and os.path.isdir(self._upload_dir):
            import shutil
            shutil.rmtree(self._upload_dir, ignore_errors=True)
        logger.info("Streaming server stopped.")


# ---------------------------------------------------------------------------
# Multipart form data parser (no external deps)
# ---------------------------------------------------------------------------

def _parse_multipart_file(data: bytes, boundary: bytes) -> Optional[bytes]:
    """
    Extract the first file field from a ``multipart/form-data`` body.

    Returns the raw file bytes, or *None* if parsing fails.
    """
    # Boundaries in the body are prefixed with --
    delim = b"--" + boundary
    parts = data.split(delim)
    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        if b'filename=' not in part:
            continue
        # Headers end at the first blank line (\r\n\r\n)
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        file_data = part[header_end + 4:]
        # Strip trailing \r\n-- (boundary suffix)
        if file_data.endswith(b"\r\n"):
            file_data = file_data[:-2]
        return file_data
    return None
