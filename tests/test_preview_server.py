import base64
import json
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from chandelier.events import Action, Phase
from chandelier.services.preview_server import PreviewServer


def _start() -> PreviewServer:
    server = PreviewServer(host="127.0.0.1", port=0)
    assert server.start() is True
    return server


def _url(server: PreviewServer, path: str) -> str:
    return f"http://127.0.0.1:{server.port}{path}"


def _get(server: PreviewServer, path: str):
    with urllib.request.urlopen(_url(server, path), timeout=2) as response:
        return response.status, response.headers, response.read()


def _post(server: PreviewServer, path: str, payload: dict) -> int:
    request = urllib.request.Request(
        _url(server, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def _read_sse_frame(response) -> bytes:
    line = response.readline()
    assert line.startswith(b"data: ")
    data = base64.b64decode(line[len(b"data: "):].strip())
    assert response.readline() == b"\n"
    return data


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _read_mjpeg_part(response) -> bytes:
    assert response.readline() == b"--frame\r\n"
    content_type = response.readline()
    assert content_type.lower().startswith(b"content-type: image/jpeg")
    length_line = response.readline()
    assert length_line.lower().startswith(b"content-length: ")
    length = int(length_line.split(b":", 1)[1])
    assert response.readline() == b"\r\n"

    data = response.read(length)
    assert response.read(2) == b"\r\n"
    # The next boundary must follow immediately, even before a newer frame.
    assert response.readline() == b"--frame\r\n"
    return data


def test_start_and_stop_are_idempotent():
    server = _start()
    try:
        assert server.enabled is True
        assert server.port > 0
        assert server.start() is True
    finally:
        server.stop()

    assert server.enabled is False
    server.stop()


def test_snapshot_serves_the_latest_frame():
    server = _start()
    try:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 255)  # red (BGR)
        server.publish(frame)

        status, headers, body = _get(server, "/snapshot.jpg")
        assert status == 200
        assert headers.get_content_type() == "image/jpeg"

        decoded = cv2.imdecode(
            np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded.shape == (24, 32, 3)
        assert int(decoded[12, 16, 2]) > 200
    finally:
        server.stop()


def test_index_page_embeds_the_stream():
    server = _start()
    try:
        status, headers, body = _get(server, "/")
        assert status == 200
        assert headers.get_content_type() == "text/html"
        assert b"/stream.mjpg" in body
    finally:
        server.stop()


def test_unknown_path_returns_404():
    server = _start()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            _get(server, "/missing")
        assert error.value.code == 404
    finally:
        server.stop()


def test_snapshot_without_a_frame_returns_503():
    server = _start()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            _get(server, "/snapshot.jpg")
        assert error.value.code == 503
    finally:
        server.stop()


def test_publish_is_ignored_while_stopped():
    server = PreviewServer(host="127.0.0.1", port=0)
    server.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    assert server.enabled is False
    assert server.snapshot() is None


def test_mjpeg_stream_contains_frames():
    server = _start()
    try:
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[:, :] = (0, 255, 0)
        server.publish(frame)

        response = urllib.request.urlopen(
            _url(server, "/stream.mjpg"), timeout=2
        )
        try:
            assert response.status == 200
            assert "multipart/x-mixed-replace" in response.headers[
                "Content-Type"
            ]
            part = _read_mjpeg_part(response)
            decoded = cv2.imdecode(
                np.frombuffer(part, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            assert decoded.shape == (16, 16, 3)
        finally:
            response.close()
    finally:
        server.stop()
    assert server.enabled is False


def test_port_in_use_keeps_the_server_stopped():
    first = _start()
    second = PreviewServer(host="127.0.0.1", port=first.port)
    try:
        assert second.start() is False
        assert second.enabled is False
        # A failed start is not retried on every menu refresh ...
        assert second.set_enabled(True) is False

        # ... but an explicit off/on cycle retries.
        second.set_enabled(False)
        first.stop()
        assert second.set_enabled(True) is True
    finally:
        second.stop()
        first.stop()


def test_sse_stream_contains_frames_and_counts_clients():
    server = _start()
    try:
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 255)
        server.publish(frame)

        response = urllib.request.urlopen(
            _url(server, "/events"), timeout=2
        )
        try:
            assert response.status == 200
            assert "text/event-stream" in response.headers["Content-Type"]
            assert _wait_for(lambda: server.client_count == 1)

            part = _read_sse_frame(response)
            decoded = cv2.imdecode(
                np.frombuffer(part, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            assert decoded.shape == (16, 16, 3)
        finally:
            response.close()
        assert _wait_for(lambda: server.client_count == 0)
    finally:
        server.stop()


def test_status_reports_server_state():
    server = _start()
    try:
        with urllib.request.urlopen(
            _url(server, "/status"), timeout=2
        ) as response:
            payload = json.loads(response.read())
        assert payload == {
            "enabled": True,
            "port": server.port,
            "clients": 0,
        }
    finally:
        server.stop()


def test_remote_control_and_screen_switch():
    server = _start()
    actions = []
    screens = []

    def on_action(action, phase):
        actions.append((action, phase))
        return True

    server.set_action_handler(on_action)
    server.set_screen_handler(screens.append)
    try:
        assert _post(server, "/control", {"action": "up"}) == 200
        assert _post(server, "/screen", {"id": "gallery"}) == 200
        assert _post(server, "/control", {"action": "nope"}) == 400
        assert _post(server, "/screen", {"id": "nope"}) == 200
    finally:
        server.stop()

    assert actions == [(Action.UP, Phase.PRESS)]
    assert screens == ["gallery", "nope"]


def test_remote_control_without_handler_fails():
    server = _start()
    try:
        assert _post(server, "/control", {"action": "up"}) == 400
        assert _post(server, "/screen", {"id": "camera"}) == 400
    finally:
        server.stop()
