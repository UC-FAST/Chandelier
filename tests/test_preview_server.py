import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np
import pytest

from chandelier.events import Action, Phase
from chandelier.media.library import MediaBrowser, source_cache_dir
from chandelier.services.preview_server import PreviewServer, load_auth_file
from chandelier.ui.themes import THEMES


def _start(
    auth: tuple[str, str] | None = None,
    media_path=None,
) -> PreviewServer:
    server = PreviewServer(
        host="127.0.0.1", port=0, auth=auth, media_path=media_path
    )
    assert server.start() is True
    return server


def _write_media(directory, name, color=(0, 0, 255), mtime=None):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :] = color
    path = directory / name
    assert cv2.imwrite(str(path), image)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _url(server: PreviewServer, path: str) -> str:
    return f"http://127.0.0.1:{server.port}{path}"


def _basic_auth(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(
        f"{username}:{password}".encode("utf-8")
    ).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _get(server: PreviewServer, path: str, headers: dict[str, str] | None = None):
    request = urllib.request.Request(
        _url(server, path), headers=headers or {}
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.status, response.headers, response.read()


def _post(
    server: PreviewServer,
    path: str,
    payload: dict,
    headers: dict[str, str] | None = None,
) -> int:
    request = urllib.request.Request(
        _url(server, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def _post_json(
    server: PreviewServer,
    path: str,
    payload: dict,
    headers: dict[str, str] | None = None,
):
    request = urllib.request.Request(
        _url(server, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, None


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


def test_touch_endpoint_calls_handler():
    server = _start()
    calls = []
    server.set_touch_handler(
        lambda x, y: calls.append((x, y)) or True
    )
    try:
        request = urllib.request.Request(
            _url(server, "/touch"),
            data=json.dumps({"x": 0.25, "y": 0.5}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        assert payload == {"ok": True}
        assert calls == [(0.25, 0.5)]
    finally:
        server.stop()


def test_touch_endpoint_rejects_invalid_coordinates():
    server = _start()
    calls = []
    server.set_touch_handler(
        lambda x, y: calls.append((x, y)) or True
    )
    try:
        request = urllib.request.Request(
            _url(server, "/touch"),
            data=json.dumps({"x": 1.5, "y": 0.5}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request, timeout=2)
        assert excinfo.value.code == 400
        assert calls == []
    finally:
        server.stop()


def test_status_reports_server_state():
    server = _start()
    try:
        with urllib.request.urlopen(
            _url(server, "/status"), timeout=2
        ) as response:
            payload = json.loads(response.read())
        assert payload["enabled"] is True
        assert payload["port"] == server.port
        assert payload["clients"] == 0
        theme = payload["theme"]
        assert theme["id"] in THEMES
        assert theme["scheme"] in {"light", "dark"}
        assert set(theme["palette"]) == {
            "bg",
            "text",
            "muted",
            "line",
            "panel",
            "panel2",
            "accent",
            "accentInk",
            "green",
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
        assert _post(
            server, "/control", {"action": "up", "phase": "release"}
        ) == 200
        assert _post(
            server, "/control", {"action": "up", "phase": "bogus"}
        ) == 400
        assert _post(server, "/screen", {"id": "gallery"}) == 200
        assert _post(server, "/control", {"action": "nope"}) == 400
        assert _post(server, "/screen", {"id": "nope"}) == 200
    finally:
        server.stop()

    assert actions == [
        (Action.UP, Phase.PRESS),
        (Action.UP, Phase.RELEASE),
    ]
    assert screens == ["gallery", "nope"]


def test_remote_control_without_handler_fails():
    server = _start()
    try:
        assert _post(server, "/control", {"action": "up"}) == 400
        assert _post(server, "/screen", {"id": "camera"}) == 400
    finally:
        server.stop()


def test_auth_file_round_trip(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps({"username": "pi", "password": "secret"}),
        encoding="utf-8",
    )
    assert load_auth_file(path) == ("pi", "secret")


def test_missing_auth_file_disables_authentication(tmp_path):
    assert load_auth_file(tmp_path / "missing.json") is None


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"username": "pi"},
        {"username": "", "password": "secret"},
        {"username": "pi", "password": ""},
        {"username": 1, "password": "secret"},
    ],
)
def test_invalid_auth_file_disables_authentication(tmp_path, payload):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_auth_file(path) is None


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/status",
        "/snapshot.jpg",
        "/events",
        "/stream.mjpg",
        "/gallery",
        "/api/gallery",
        "/thumb/a.jpg",
        "/view/a.jpg",
        "/media/a.jpg",
    ],
)
def test_auth_rejects_unauthenticated_gets(path):
    server = _start(("pi", "secret"))
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            _get(server, path)
        assert error.value.code == 401
        assert error.value.headers["WWW-Authenticate"].startswith("Basic ")
    finally:
        server.stop()


def test_auth_rejects_wrong_credentials_and_malformed_headers():
    server = _start(("pi", "secret"))
    try:
        for headers in (
            _basic_auth("pi", "wrong"),
            _basic_auth("wrong", "secret"),
            {"Authorization": "Basic !!!"},
            {"Authorization": "Bearer secret"},
        ):
            with pytest.raises(urllib.error.HTTPError) as error:
                _get(server, "/status", headers)
            assert error.value.code == 401
    finally:
        server.stop()


def test_auth_protects_post_endpoints():
    server = _start(("pi", "secret"))
    try:
        assert _post(server, "/control", {"action": "up"}) == 401
        assert _post(server, "/touch", {"x": 0.5, "y": 0.5}) == 401
        assert (
            _post(
                server,
                "/api/gallery/delete",
                {"ids": ["a.jpg"]},
            )
            == 401
        )
    finally:
        server.stop()


def test_auth_allows_authenticated_requests():
    server = _start(("pi", "secret"))
    server.set_action_handler(lambda action, phase: True)
    try:
        status, headers, body = _get(
            server, "/status", _basic_auth("pi", "secret")
        )
        assert status == 200
        assert json.loads(body)["enabled"] is True

        assert (
            _post(
                server,
                "/control",
                {"action": "up"},
                _basic_auth("pi", "secret"),
            )
            == 200
        )
    finally:
        server.stop()


def test_gallery_page_and_api_list_media(tmp_path):
    _write_media(tmp_path, "older.jpg", (0, 0, 255), mtime=1_000_000)
    _write_media(tmp_path, "newer.jpg", (0, 255, 0), mtime=2_000_000)
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    server = _start(media_path=tmp_path)
    try:
        status, headers, body = _get(server, "/gallery")
        assert status == 200
        assert headers.get_content_type() == "text/html"
        assert b"/api/gallery" in body

        payload = json.loads(_get(server, "/api/gallery")[2])
        assert payload["count"] == 2
        assert [item["name"] for item in payload["items"]] == [
            "newer.jpg",
            "older.jpg",
        ]
        assert payload["items"][0]["type"] == "image"
        assert payload["items"][0]["id"] == payload["items"][0]["name"]
        assert payload["items"][0]["version"]
        assert payload["items"][0]["size"] > 0
    finally:
        server.stop()


def test_gallery_scans_photo_and_video_directories(tmp_path):
    photos = tmp_path / "photos"
    videos = tmp_path / "videos"
    photos.mkdir()
    videos.mkdir()
    _write_media(photos, "photo.jpg")
    video_bytes = b"fake-mp4-payload"
    (videos / "clip.mp4").write_bytes(video_bytes)

    server = _start(media_path=[photos, videos])
    try:
        payload = json.loads(_get(server, "/api/gallery")[2])
        by_id = {item["id"]: item for item in payload["items"]}
        assert set(by_id) == {"photo.jpg", "videos/clip.mp4"}
        assert by_id["videos/clip.mp4"]["name"] == "clip.mp4"
        assert by_id["videos/clip.mp4"]["type"] == "video"

        item_id = urllib.parse.quote("videos/clip.mp4", safe="")
        request = urllib.request.Request(
            _url(server, f"/media/{item_id}"),
            headers={"Range": "bytes=0-3"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 206
            assert response.headers.get_content_type() == "video/mp4"
            assert response.read() == video_bytes[:4]

        with pytest.raises(urllib.error.HTTPError) as error:
            _get(server, f"/thumb/{item_id}?w=64&h=64")
        assert error.value.code == 404
    finally:
        server.stop()


def test_gallery_view_renders_tiff_as_jpeg(tmp_path):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 255)
    assert cv2.imwrite(str(tmp_path / "photo.tiff"), image)
    original = _write_media(tmp_path, "photo.jpg")

    server = _start(media_path=tmp_path)
    try:
        status, headers, body = _get(server, "/view/photo.tiff")
        assert status == 200
        assert headers.get_content_type() == "image/jpeg"
        decoded = cv2.imdecode(
            np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded.shape == (120, 160, 3)

        _, headers, body = _get(server, "/view/photo.jpg")
        assert headers.get_content_type() == "image/jpeg"
        assert body == original.read_bytes()
    finally:
        server.stop()


def test_gallery_with_missing_media_directory_is_empty(tmp_path):
    server = _start(media_path=tmp_path / "missing")
    try:
        payload = json.loads(_get(server, "/api/gallery")[2])
        assert payload == {"count": 0, "items": []}
        assert _get(server, "/gallery")[0] == 200
    finally:
        server.stop()


def test_gallery_thumbnail_is_cover_cropped_jpeg(tmp_path):
    _write_media(tmp_path, "photo.jpg")
    server = _start(media_path=tmp_path)
    try:
        name = urllib.parse.quote("photo.jpg")
        status, headers, body = _get(server, f"/thumb/{name}?w=64&h=64")
        assert status == 200
        assert headers.get_content_type() == "image/jpeg"
        decoded = cv2.imdecode(
            np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded.shape == (64, 64, 3)
    finally:
        server.stop()


def test_gallery_thumbnail_rejects_videos_and_unknown_names(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"not really a video")
    server = _start(media_path=tmp_path)
    try:
        payload = json.loads(_get(server, "/api/gallery")[2])
        assert payload["count"] == 1
        assert payload["items"][0]["type"] == "video"

        for path in (
            "/thumb/clip.mp4",
            "/thumb/missing.jpg",
            "/media/missing.jpg",
        ):
            with pytest.raises(urllib.error.HTTPError) as error:
                _get(server, path)
            assert error.value.code == 404
    finally:
        server.stop()


def test_gallery_media_serves_original_bytes_and_ranges(tmp_path):
    path = _write_media(tmp_path, "photo.jpg")
    original = path.read_bytes()
    server = _start(media_path=tmp_path)
    try:
        name = urllib.parse.quote("photo.jpg")
        status, headers, body = _get(server, f"/media/{name}")
        assert status == 200
        assert headers.get_content_type() == "image/jpeg"
        assert headers["Accept-Ranges"] == "bytes"
        assert "filename*=UTF-8''photo.jpg" in headers[
            "Content-Disposition"
        ]
        assert body == original

        request = urllib.request.Request(
            _url(server, f"/media/{name}"),
            headers={"Range": "bytes=10-19"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == (
                f"bytes 10-19/{len(original)}"
            )
            assert response.read() == original[10:20]

        # Unknown range units are ignored, not rejected.
        request = urllib.request.Request(
            _url(server, f"/media/{name}"),
            headers={"Range": "items=0-1"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 200
            assert response.read() == original
    finally:
        server.stop()


def test_gallery_media_rejects_invalid_ranges(tmp_path):
    _write_media(tmp_path, "photo.jpg")
    server = _start(media_path=tmp_path)
    try:
        name = urllib.parse.quote("photo.jpg")
        request = urllib.request.Request(
            _url(server, f"/media/{name}"),
            headers={"Range": "bytes=999999-"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=2)
        assert error.value.code == 416
        assert error.value.headers["Content-Range"].startswith("bytes */")
    finally:
        server.stop()


def test_gallery_media_serves_empty_files_cleanly(tmp_path):
    (tmp_path / "empty.jpg").write_bytes(b"")
    server = _start(media_path=tmp_path)
    try:
        name = urllib.parse.quote("empty.jpg")
        status, headers, body = _get(server, f"/media/{name}")
        assert status == 200
        assert headers["Content-Length"] == "0"
        assert body == b""
    finally:
        server.stop()


def test_gallery_ignores_non_utf8_names(tmp_path):
    path = os.path.join(os.fsencode(tmp_path), b"bad\xff.jpg")
    with open(path, "wb") as handle:
        handle.write(b"not an image")
    server = _start(media_path=tmp_path)
    try:
        payload = json.loads(_get(server, "/api/gallery")[2])
        assert payload == {"count": 0, "items": []}
    finally:
        server.stop()


def test_gallery_does_not_follow_symlinks_swapped_after_scan(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    target = _write_media(media, "swap.jpg")
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"SECRET")
    server = _start(media_path=media)
    try:
        assert json.loads(_get(server, "/api/gallery")[2])["count"] == 1

        target.unlink()
        target.symlink_to(secret)
        name = urllib.parse.quote("swap.jpg")
        for path in (
            f"/media/{name}",
            f"/thumb/{name}?w=64&h=64",
        ):
            with pytest.raises(urllib.error.HTTPError) as error:
                _get(server, path)
            assert error.value.code == 404
    finally:
        server.stop()


def test_gallery_thumbnail_stops_serving_deleted_files(tmp_path):
    path = _write_media(tmp_path, "gone.jpg")
    server = _start(media_path=tmp_path)
    try:
        name = urllib.parse.quote("gone.jpg")
        assert _get(server, f"/thumb/{name}?w=64&h=64")[0] == 200

        path.unlink()
        with pytest.raises(urllib.error.HTTPError) as error:
            _get(server, f"/thumb/{name}?w=64&h=64")
        assert error.value.code == 404
    finally:
        server.stop()


def test_gallery_thumbnail_handles_extreme_aspect_ratios(tmp_path):
    image = np.zeros((8, 4096, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "strip.jpg"), image)
    server = _start(media_path=tmp_path)
    try:
        name = urllib.parse.quote("strip.jpg")
        status, _, body = _get(server, f"/thumb/{name}?w=400&h=400")
        assert status == 200
        decoded = cv2.imdecode(
            np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded.shape == (512, 512, 3)
    finally:
        server.stop()


def test_gallery_delete_removes_files_caches_and_thumbnails(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _write_media(tmp_path, "a.jpg")
    _write_media(tmp_path, "b.jpg")

    browser = MediaBrowser(str(tmp_path), 320, 240)
    index = next(
        i
        for i in range(browser.total_media())
        if os.path.basename(
            browser.get_media_info(i)["path"]
        ) == "a.jpg"
    )
    browser.jump_to_index(index)
    assert browser.get_current_frame() is not None
    cache_dir = source_cache_dir(str(tmp_path / "a.jpg"))
    assert os.path.isdir(cache_dir)

    server = _start(media_path=tmp_path)
    try:
        assert _get(server, "/thumb/a.jpg?w=64&h=64")[0] == 200

        status, payload = _post_json(
            server, "/api/gallery/delete", {"ids": ["a.jpg"]}
        )
        assert status == 200
        assert payload == {
            "ok": True,
            "deleted": ["a.jpg"],
            "failed": [],
        }
        assert not (tmp_path / "a.jpg").exists()
        assert (tmp_path / "b.jpg").exists()
        assert not os.path.exists(cache_dir)

        listing = json.loads(
            _get(server, "/api/gallery?refresh=1")[2]
        )
        assert [item["name"] for item in listing["items"]] == ["b.jpg"]

        with pytest.raises(urllib.error.HTTPError) as error:
            _get(server, "/thumb/a.jpg?w=64&h=64")
        assert error.value.code == 404
    finally:
        server.stop()


def test_gallery_delete_reports_unknown_and_traversal_ids(tmp_path):
    _write_media(tmp_path, "a.jpg")
    server = _start(media_path=tmp_path)
    try:
        status, payload = _post_json(
            server,
            "/api/gallery/delete",
            {"ids": ["missing.jpg", "../secret.jpg"]},
        )
        assert status == 200
        assert payload["deleted"] == []
        assert set(payload["failed"]) == {
            "missing.jpg",
            "../secret.jpg",
        }
        assert (tmp_path / "a.jpg").exists()

        status, _ = _post_json(
            server, "/api/gallery/delete", {"ids": []}
        )
        assert status == 400
    finally:
        server.stop()


def test_gallery_without_media_is_empty(tmp_path):
    server = _start(media_path=tmp_path)
    try:
        payload = json.loads(_get(server, "/api/gallery")[2])
        assert payload == {"count": 0, "items": []}

        status, _, body = _get(server, "/gallery")
        assert status == 200
        assert b"/api/gallery" in body
    finally:
        server.stop()


def test_gallery_refresh_picks_up_new_files(tmp_path):
    server = _start(media_path=tmp_path)
    try:
        assert json.loads(_get(server, "/api/gallery")[2])["count"] == 0
        _write_media(tmp_path, "late.jpg")
        payload = json.loads(
            _get(server, "/api/gallery?refresh=1")[2]
        )
        assert payload["count"] == 1
        assert payload["items"][0]["name"] == "late.jpg"
    finally:
        server.stop()


def test_gallery_rejects_traversal_names(tmp_path):
    (tmp_path.parent / "secret.jpg").write_bytes(b"secret")
    server = _start(media_path=tmp_path)
    try:
        for path in (
            "/media/..%2Fsecret.jpg",
            "/thumb/..%2Fsecret.jpg",
        ):
            with pytest.raises(urllib.error.HTTPError) as error:
                _get(server, path)
            assert error.value.code == 404
    finally:
        server.stop()


def test_gallery_ignores_symlinks_outside_the_media_directory(tmp_path):
    outside = _write_media(tmp_path, "outside.jpg")
    media = tmp_path / "media"
    media.mkdir()
    (media / "link.jpg").symlink_to(outside)
    server = _start(media_path=media)
    try:
        payload = json.loads(_get(server, "/api/gallery")[2])
        assert payload == {"count": 0, "items": []}
    finally:
        server.stop()


def test_index_page_maps_all_physical_buttons():
    server = _start()
    try:
        _, _, body = _get(server, "/")
    finally:
        server.stop()
    html = body.decode("utf-8")
    for action in (
        "up",
        "down",
        "left",
        "right",
        "center",
        "circle",
        "square",
        "cross",
        "shutter",
        "rotary_clockwise",
        "rotary_counter_clockwise",
        "rotary_select",
    ):
        assert f'data-action="{action}"' in html
