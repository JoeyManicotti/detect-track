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

import json
import logging
import socketserver
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

_INDEX_HTML = b"""\
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
#video-wrap{flex:1;display:flex;align-items:center;justify-content:center;
            background:#000;overflow:hidden;position:relative}
#stream{max-width:100%;max-height:100%;display:block;object-fit:contain}
#stream-status{position:absolute;top:10px;left:10px;font-size:11px;
               background:rgba(0,0,0,.65);padding:3px 9px;border-radius:4px;color:#0f0}
#stream-status.disconnected{color:#f44}
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
<div id="video-wrap">
  <img id="stream" src="/video" alt="video stream">
  <div id="stream-status">\u25cf LIVE</div>
</div>
<div id="sidebar">
  <div id="sidebar-header">detect-track</div>
  <div id="stats-bar">
    <span>frame <span class="sv" id="sf">\u2013</span></span>
    <span>tracks <span class="sv" id="sc">0</span></span>
  </div>
  <div id="tracks-el"><div class="empty">Waiting for detections\u2026<br>Ensure the pipeline is running.</div></div>
</div>
<script>
// colour palette (golden-ratio hue spacing, no external deps)
const cc={};
function hsv2rgb(h,s,v){const i=Math.floor(h*6),f=h*6-i,p=v*(1-s),q=v*(1-f*s),t=v*(1-(1-f)*s);
  return[[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i%6].map(x=>~~(x*255))}
function col(id){if(!cc[id]){const[r,g,b]=hsv2rgb((id*0.6180339887)%1,.85,.95);
  cc[id]=`rgb(${r},${g},${b})`}return cc[id]}

// stream reconnect
const si=document.getElementById('stream'),ss=document.getElementById('stream-status');
function conn(){si.src='/video?'+Date.now();ss.textContent='\u25cf LIVE';ss.className=''}
si.onerror=()=>{ss.textContent='\u25cb reconnecting\u2026';ss.className='disconnected';setTimeout(conn,2000)};

// track polling
const sf=document.getElementById('sf'),sc=document.getElementById('sc'),te=document.getElementById('tracks-el');
async function poll(){
  try{
    const d=await(await fetch('/tracks',{cache:'no-store'})).json();
    sf.textContent=d.frame_id; sc.textContent=d.tracks.length;
    if(!d.tracks.length){te.innerHTML='<div class="empty">No active tracks</div>'}
    else te.innerHTML=d.tracks.map(t=>{
      const c=col(t.id),b=t.box?`[${t.box.map(v=>~~v).join(', ')}]`:'\u2013';
      return`<div class="card" style="border-left-color:${c}">
        <div class="row"><span class="tid" style="color:${c}">#${t.id}</span>
        <span class="score">${(t.score*100).toFixed(1)}%</span></div>
        <div class="label">${t.label}</div><div class="box">${b}</div></div>`}).join('');
  }catch(_){}
  setTimeout(poll,250);
}
poll();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Threaded HTTP server
# ---------------------------------------------------------------------------

class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Each client connection runs in its own thread."""
    daemon_threads = True
    allow_reuse_address = True


class StreamingServer:
    """
    Lightweight MJPEG / JSON streaming server.

    Usage::

        server = StreamingServer(host="0.0.0.0", port=8080)
        server.start()                     # non-blocking

        for result in pipeline.results():
            frame_bgr = annotate(frame, result)
            server.push_frame(frame_bgr, result)

        server.stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = port

        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._frame_id: int = 0
        self._tracks: List[dict] = []

        self._server: Optional[_ThreadedHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_frame(
        self,
        frame_bgr: np.ndarray,
        result: Optional[TrackResult] = None,
        *,
        jpeg_quality: int = 80,
    ) -> None:
        """
        Encode *frame_bgr* as JPEG and update the stream.

        Parameters
        ----------
        frame_bgr:
            BGR uint8 frame to broadcast.
        result:
            If provided, the track data is exposed on ``/tracks``.
        jpeg_quality:
            JPEG encoding quality [1, 100].
        """
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

        with self._lock:
            self._jpeg = buf.tobytes()
            self._frame_id += 1
            self._tracks = tracks

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        streaming = self  # captured by handler class

        class _Handler(BaseHTTPRequestHandler):
            # Silence per-request access log noise.
            def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
                pass

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?")[0]
                if path == "/video":
                    self._mjpeg()
                elif path == "/tracks":
                    self._json_tracks()
                elif path in ("/", "/index.html"):
                    self._html()
                elif path == "/health":
                    self._text(b"ok")
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

            # ── Plain text helper ──────────────────────────────────
            def _text(self, body: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
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
        logger.info("Streaming server stopped.")
