"""MJPEG preview server for the device screen.

The controller publishes every fully composed frame (camera, menu, gallery,
system monitor) into :class:`PreviewServer`.  A small standard-library HTTP
server exposes the latest frame as an MJPEG stream so a browser on the local
network sees the same picture as the LCD.
"""

from __future__ import annotations

import base64
import json
import logging
import select
import socket
import threading
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2

from chandelier.events import Action, Phase
from chandelier.i18n import tr, translator

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5989
JPEG_QUALITY = 80

_INDEX_TEMPLATE = """<!doctype html>
<html lang="{{LANG}}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0d1117">
<title>Chandelier Preview</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    min-height: 100vh; min-height: 100dvh;
    background: radial-gradient(1200px 600px at 50% -10%,
                #1b2430 0%, #0d1117 55%, #090c10 100%);
    color: #e6edf3;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    -webkit-tap-highlight-color: transparent;
    overscroll-behavior: none;
  }
  .app {
    min-height: 100vh; min-height: 100dvh;
    display: flex; flex-direction: column; align-items: center;
    justify-content: flex-start; gap: 10px; padding: 10px;
  }
  .preview {
    width: 100%; display: flex; flex-direction: column;
    align-items: center; gap: 6px; flex: 0 0 auto;
  }
  .frame {
    width: min(100%, 640px); padding: 4px;
    background: linear-gradient(180deg, #1c2128, #11151c);
    border: 1px solid #30363d; border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, .45);
  }
  img {
    display: block; width: 100%; height: auto; aspect-ratio: 4 / 3;
    object-fit: contain; image-rendering: pixelated;
    background: #000; border-radius: 10px;
  }
  .meta { margin: 0; font-size: .78rem; color: #8b949e; text-align: center; }
  .meta a { color: #58a6ff; text-decoration: none; }
  .controls {
    width: 100%; max-width: 640px;
    display: flex; flex-direction: column; gap: 10px;
  }
  .screens { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .screens button {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 4px; padding: 10px 4px;
    font-size: .75rem;
  }
  .pad {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
    grid-template-areas: ". up ." "left ok right" "back down shot";
  }
  button {
    font: inherit; color: #e6edf3;
    background: linear-gradient(180deg, #21262d, #161b22);
    border: 1px solid #30363d; border-radius: 14px;
    cursor: pointer; touch-action: manipulation; user-select: none;
    min-height: 52px;
    transition: transform .06s ease, background .15s ease,
                border-color .15s ease;
  }
  button:hover { border-color: #4a5561; }
  button:active { transform: scale(.96); background: #30363d; }
  button:focus-visible { outline: 2px solid #58a6ff; outline-offset: 2px; }
  .icon { width: 22px; height: 22px; fill: currentColor; }
  .dpad .icon { width: 30px; height: 30px; }
  .up .icon { transform: rotate(0deg); }
  .right .icon { transform: rotate(90deg); }
  .down .icon { transform: rotate(180deg); }
  .left .icon { transform: rotate(-90deg); }
  .ok {
    font-weight: 700; letter-spacing: .04em; color: #fff;
    background: linear-gradient(180deg, #1f6feb, #1158c7);
    border-color: #388bfd;
  }
  .back, .shot {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 2px; font-size: .8rem;
  }
  .up { grid-area: up; } .down { grid-area: down; }
  .left { grid-area: left; } .right { grid-area: right; }
  .ok { grid-area: ok; } .back { grid-area: back; }
  .shot { grid-area: shot; }
  @media (orientation: landscape) {
    .app { padding: 6px; gap: 6px; }
    .frame { width: min(100%, 640px, calc(46dvh * 4 / 3)); }
    .controls {
      flex-direction: row; align-items: stretch;
      max-width: min(100%, 760px); gap: 10px;
    }
    .screens { grid-template-columns: repeat(2, 1fr); flex: 1 1 0; }
    .pad { flex: 1 1 0; }
    button { min-height: 44px; }
    .screens button { padding: 6px 4px; }
    .dpad .icon { width: 24px; height: 24px; }
  }
</style>
</head>
<body>
<main class="app">
  <section class="preview">
    <div class="frame"><img id="live" alt="{{LIVE_PREVIEW}}" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"></div>
    <p class="meta"><span id="status">{{CONNECTING}}</span> &middot;
       320 &times; 240 &middot; <a href="/snapshot.jpg">{{SNAPSHOT}}</a></p>
  </section>
  <section class="controls" aria-label="Controls">
    <nav class="screens">
      <button data-screen="camera" aria-label="{{CAMERA}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l1.5-2h3L15 5h4a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4zm3 4.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9zm0 2a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5z"/></svg>{{CAMERA}}</button>
      <button data-screen="gallery" aria-label="{{GALLERY}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zm1 2v10h14V7H5zm2 8l3-4 2.5 3 2-2.5L18 15H7z"/><circle cx="8.5" cy="9.5" r="1.3"/></svg>{{GALLERY}}</button>
      <button data-screen="menu" aria-label="{{MENU}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16v2H4zm0 5h16v2H4zm0 5h16v2H4z"/></svg>{{MENU}}</button>
      <button data-screen="system_monitor" aria-label="{{SYSTEM}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 4h18a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1h-7v2h3v2H7v-2h3v-2H3a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1zm1 2v9h16V6H4z"/></svg>{{SYSTEM}}</button>
    </nav>
    <div class="pad">
      <button class="dpad up" data-action="up" aria-label="{{UP}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l-7 8h4v8h6v-8h4z"/></svg></button>
      <button class="dpad left" data-action="left" aria-label="{{LEFT}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l-7 8h4v8h6v-8h4z"/></svg></button>
      <button class="ok" data-action="center" aria-label="{{OK}}">{{OK}}</button>
      <button class="dpad right" data-action="right" aria-label="{{RIGHT}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l-7 8h4v8h6v-8h4z"/></svg></button>
      <button class="back" data-action="cross" aria-label="{{BACK}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M10 5l-7 7 7 7v-4h5a4 4 0 0 1 4 4h2a6 6 0 0 0-6-6h-5z"/></svg>{{BACK}}</button>
      <button class="dpad down" data-action="down" aria-label="{{DOWN}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l-7 8h4v8h6v-8h4z"/></svg></button>
      <button class="shot" data-action="shutter" aria-label="{{SHOT}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="4"/></svg>{{SHOT}}</button>
    </div>
  </section>
</main>
<script>
const post = (path, body) => fetch(path, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(body),
}).catch(() => {});
document.querySelectorAll('[data-action]').forEach((button) => {
  button.addEventListener('click', () => {
    post('/control', {action: button.dataset.action});
  });
});
document.querySelectorAll('[data-screen]').forEach((button) => {
  button.addEventListener('click', () => {
    post('/screen', {id: button.dataset.screen});
  });
});

const live = document.getElementById('live');
const status = document.getElementById('status');
if (window.EventSource) {
  const source = new EventSource('/events');
  source.onopen = () => { status.textContent = {{STREAM_CONNECTED}}; };
  source.onerror = () => { status.textContent = {{RECONNECTING}}; };
  source.onmessage = (event) => {
    live.src = 'data:image/jpeg;base64,' + event.data;
  };
} else {
  live.src = '/stream.mjpg';
  status.textContent = {{MJPEG}};
}
</script>
</body>
</html>
"""


def _index_html() -> bytes:
    values = {
        "LANG": escape(translator.language),
        "LIVE_PREVIEW": escape(tr("Live preview")),
        "CAMERA": escape(tr("Camera")),
        "GALLERY": escape(tr("Gallery")),
        "MENU": escape(tr("Menu")),
        "SYSTEM": escape(tr("System")),
        "UP": escape(tr("Up")),
        "LEFT": escape(tr("Left")),
        "OK": escape(tr("OK")),
        "RIGHT": escape(tr("Right")),
        "BACK": escape(tr("Back")),
        "DOWN": escape(tr("Down")),
        "SHOT": escape(tr("Shot")),
        "CONNECTING": escape(tr("connecting...")),
        "SNAPSHOT": escape(tr("snapshot")),
        "STREAM_CONNECTED": json.dumps(tr("stream connected")),
        "RECONNECTING": json.dumps(tr("reconnecting...")),
        "MJPEG": json.dumps("mjpeg"),
    }
    html = _INDEX_TEMPLATE
    for token, value in values.items():
        html = html.replace("{{" + token + "}}", value)
    return html.encode("utf-8")


class PreviewServer:
    """Publish frames and serve them as an MJPEG stream."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ):
        self._host = host
        self._port = int(port)
        self._state_lock = threading.Lock()
        self._condition = threading.Condition()
        self._encode_lock = threading.Lock()
        self._enabled = False
        self._start_failed = False
        self._server: _PreviewHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._frame = None
        self._sequence = 0
        self._jpeg: bytes | None = None
        self._jpeg_sequence = -1
        self._stopping = threading.Event()
        self._action_handler = None
        self._screen_handler = None
        self._clients = 0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def port(self) -> int:
        server = self._server
        if server is not None:
            return int(server.server_address[1])
        return self._port

    @property
    def stop_event(self) -> threading.Event:
        """Event that is set when the current server generation stops."""

        return self._stopping

    def set_enabled(self, enabled: bool) -> bool:
        if enabled:
            if self._start_failed:
                return False
            return self.start()
        self._start_failed = False
        self.stop()
        return True

    def set_action_handler(self, handler) -> None:
        """Register ``handler(Action, Phase) -> bool`` for remote buttons."""

        self._action_handler = handler

    def set_screen_handler(self, handler) -> None:
        """Register ``handler(end_id)`` for remote screen switching."""

        self._screen_handler = handler

    def send_action(self, name: str) -> bool:
        handler = self._action_handler
        if handler is None:
            return False
        try:
            action = Action(name)
        except ValueError:
            return False
        try:
            return bool(handler(action, Phase.PRESS))
        except Exception:
            logger.exception("preview action %r failed", name)
            return False

    def switch_to(self, screen_id: str) -> bool:
        handler = self._screen_handler
        if handler is None:
            return False
        try:
            handler(screen_id)
        except (LookupError, ValueError):
            return False
        except Exception:
            logger.exception("preview screen switch failed")
            return False
        return True

    @property
    def client_count(self) -> int:
        with self._condition:
            return self._clients

    def client_open(self) -> None:
        with self._condition:
            self._clients += 1

    def client_close(self) -> None:
        with self._condition:
            self._clients = max(0, self._clients - 1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        with self._state_lock:
            if self._enabled:
                return True
            stopping = threading.Event()
            try:
                server = _PreviewHTTPServer(
                    (self._host, self._port), _PreviewHandler
                )
            except OSError:
                logger.warning(
                    "preview server could not bind %s:%s",
                    self._host,
                    self._port,
                    exc_info=True,
                )
                self._start_failed = True
                return False

            server.preview = self
            self._stopping = stopping
            with self._condition:
                self._frame = None
                self._jpeg = None
                self._jpeg_sequence = -1
                self._clients = 0
            self._server = server
            self._thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="preview-server",
                daemon=True,
            )
            self._enabled = True
            self._start_failed = False
            self._thread.start()

        logger.info(
            "preview server listening on http://%s:%s/",
            self._host,
            self.port,
        )
        return True

    def stop(self) -> None:
        with self._state_lock:
            if not self._enabled and self._server is None:
                return
            server = self._server
            thread = self._thread
            stopping = self._stopping
            self._server = None
            self._thread = None
            self._enabled = False

        stopping.set()
        with self._condition:
            self._condition.notify_all()

        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)

        with self._condition:
            self._frame = None
            self._jpeg = None
            self._jpeg_sequence = -1
            self._sequence += 1
            self._condition.notify_all()

        logger.info("preview server stopped")

    # ------------------------------------------------------------------
    # Frame publishing
    # ------------------------------------------------------------------
    def publish(self, frame) -> None:
        """Publish one composed BGR frame (no-op when disabled)."""

        if not self._enabled or frame is None:
            return
        snapshot = frame.copy()
        with self._condition:
            if not self._enabled:
                return
            self._frame = snapshot
            self._sequence += 1
            self._condition.notify_all()

    def latest_frame(self):
        with self._condition:
            if self._frame is None:
                return None
            return self._sequence, self._frame

    def jpeg_for(self, sequence: int, frame) -> bytes | None:
        """Encode ``frame`` once per sequence and share it with all clients."""

        with self._encode_lock:
            with self._condition:
                if (
                    self._jpeg is not None
                    and self._jpeg_sequence == sequence
                ):
                    return self._jpeg
            try:
                ok, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                )
            except cv2.error:
                logger.debug("failed to encode preview frame", exc_info=True)
                return None
            if not ok:
                return None

            jpeg = buffer.tobytes()
            with self._condition:
                if sequence >= self._jpeg_sequence:
                    self._jpeg = jpeg
                    self._jpeg_sequence = sequence
            return jpeg

    def snapshot(self) -> bytes | None:
        latest = self.latest_frame()
        if latest is None:
            return None
        sequence, frame = latest
        return self.jpeg_for(sequence, frame)

    def wait_for_frame(
        self, after: int, timeout: float = 1.0
    ):
        with self._condition:
            if self._sequence == after or self._frame is None:
                self._condition.wait(timeout)
            if self._frame is None:
                return None
            return self._sequence, self._frame


class _PreviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    preview: PreviewServer


class _PreviewHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 10
    disable_nagle_algorithm = True

    def log_message(self, format, *args) -> None:
        logger.debug("preview http: " + format, *args)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in {"", "/"}:
            self._send_body(_index_html(), "text/html; charset=utf-8")
        elif path in {"/stream.mjpg", "/stream"}:
            self._stream()
        elif path == "/events":
            self._events()
        elif path == "/status":
            self._status()
        elif path in {"/snapshot.jpg", "/snapshot"}:
            self._snapshot()
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/control":
            self._control()
        elif path == "/screen":
            self._screen()
        else:
            self.send_error(404, "Not Found")

    def _send_body(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def _snapshot(self) -> None:
        jpeg = self.server.preview.snapshot()
        if jpeg is None:
            self.send_error(503, "No frame available")
            return
        self._send_body(jpeg, "image/jpeg")

    def _client_gone(self) -> bool:
        """Detect a closed peer without writing (used while idle)."""

        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
        except (OSError, ValueError):
            return True
        if not readable:
            return False
        try:
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _stream(self) -> None:
        preview = self.server.preview
        stopping = preview.stop_event
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store, no-cache")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        preview.client_open()
        sequence = -1
        first = True
        try:
            while not stopping.is_set():
                item = preview.wait_for_frame(sequence)
                if item is None:
                    if self._client_gone():
                        break
                    continue
                current, frame = item
                if current == sequence:
                    if self._client_gone():
                        break
                    continue
                jpeg = preview.jpeg_for(current, frame)
                if jpeg is None:
                    continue
                sequence = current

                if first:
                    self.wfile.write(b"--frame\r\n")
                    first = False
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(jpeg)
                # Close the part immediately.  MJPEG parsers delimit a frame
                # by the next boundary, so without this a static menu frame
                # stays buffered in the browser until another frame arrives.
                self.wfile.write(b"\r\n--frame\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass
        finally:
            preview.client_close()

    def _events(self) -> None:
        """Server-Sent Events stream: one base64 JPEG per event."""

        preview = self.server.preview
        stopping = preview.stop_event
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        preview.client_open()
        sequence = -1
        try:
            while not stopping.is_set():
                item = preview.wait_for_frame(sequence)
                if item is None:
                    if self._client_gone():
                        break
                    continue
                current, frame = item
                if current == sequence:
                    if self._client_gone():
                        break
                    continue
                jpeg = preview.jpeg_for(current, frame)
                if jpeg is None:
                    continue
                sequence = current
                self.wfile.write(b"data: ")
                self.wfile.write(base64.b64encode(jpeg))
                self.wfile.write(b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass
        finally:
            preview.client_close()

    def _status(self) -> None:
        preview = self.server.preview
        self._json_response(
            200,
            {
                "enabled": preview.enabled,
                "port": preview.port,
                "clients": preview.client_count,
            },
        )

    def _control(self) -> None:
        data = self._read_json()
        action = data.get("action") if isinstance(data, dict) else None
        ok = bool(action) and self.server.preview.send_action(str(action))
        self._json_response(200 if ok else 400, {"ok": ok})

    def _screen(self) -> None:
        data = self._read_json()
        screen = data.get("id") if isinstance(data, dict) else None
        ok = bool(screen) and self.server.preview.switch_to(str(screen))
        self._json_response(200 if ok else 400, {"ok": ok})

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            return None
        if not 0 < length <= 4096:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except (TypeError, ValueError):
            return None

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)
