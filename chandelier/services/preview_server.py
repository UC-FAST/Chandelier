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
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chandelier Preview</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; min-height: 100vh; display: flex; flex-direction: column;
         align-items: center; justify-content: center; gap: 12px;
         background: #111; color: #eee;
         font-family: system-ui, sans-serif; }
  h1 { margin: 0; font-size: 1.1rem; font-weight: 500; letter-spacing: .08em; }
  .frame { padding: 4px; background: #222; border: 1px solid #444;
           border-radius: 6px; }
  img { display: block; width: min(90vw, 640px); height: auto;
        image-rendering: pixelated; background: #000; }
  p { margin: 0; font-size: .8rem; color: #888; }
  a { color: #6cf; }
  .controls { width: min(90vw, 640px); display: flex; flex-direction: column;
              gap: 6px; }
  .screens { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
  .pad { display: grid; gap: 6px;
         grid-template-areas: ". up ." "left ok right"
                              "back down shot"; }
  button { padding: 10px 6px; font: inherit; font-size: .85rem;
           color: #eee; background: #263238; border: 1px solid #455a64;
           border-radius: 6px; cursor: pointer; touch-action: manipulation; }
  button:active { background: #455a64; }
  .up { grid-area: up; } .down { grid-area: down; }
  .left { grid-area: left; } .right { grid-area: right; }
  .ok { grid-area: ok; } .back { grid-area: back; }
  .shot { grid-area: shot; }
</style>
</head>
<body>
<h1>CHANDELIER &middot; LIVE</h1>
<div class="frame"><img id="live" alt="{{LIVE_PREVIEW}}"></div>
<div class="controls">
  <div class="screens">
    <button data-screen="camera">{{CAMERA}}</button>
    <button data-screen="gallery">{{GALLERY}}</button>
    <button data-screen="menu">{{MENU}}</button>
    <button data-screen="system_monitor">{{SYSTEM}}</button>
  </div>
  <div class="pad">
    <button class="up" data-action="up">{{UP}}</button>
    <button class="left" data-action="left">{{LEFT}}</button>
    <button class="ok" data-action="center">{{OK}}</button>
    <button class="right" data-action="right">{{RIGHT}}</button>
    <button class="back" data-action="cross">{{BACK}}</button>
    <button class="down" data-action="down">{{DOWN}}</button>
    <button class="shot" data-action="shutter">{{SHOT}}</button>
  </div>
</div>
<p><span id="status">{{CONNECTING}}</span> &middot; 320 &times; 240 &middot;
   <a href="/snapshot.jpg">{{SNAPSHOT}}</a></p>
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
