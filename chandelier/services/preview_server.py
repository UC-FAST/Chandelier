"""MJPEG preview server for the device screen.

The controller publishes every fully composed frame (camera, menu, gallery,
system monitor) into :class:`PreviewServer`.  A small standard-library HTTP
server exposes the latest frame as an MJPEG stream so a browser on the local
network sees the same picture as the LCD.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import mimetypes
import os
import select
import socket
import threading
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

import cv2

from chandelier.events import Action, Phase
from chandelier.i18n import tr, translator
from chandelier.services.web_gallery import (
    BROWSER_IMAGE_EXTENSIONS,
    WebGallery,
)
from chandelier.ui.themes import active_theme_id, get_theme

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5989
JPEG_QUALITY = 92
_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
if (
    hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR")
    and hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444")
):
    # Keep full chroma resolution so coloured menu text stays sharp.
    _JPEG_PARAMS += [
        int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
        int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444),
    ]


def _parse_byte_range(header: str, size: int) -> tuple[int, int] | None:
    """Parse the first range of a ``Range: bytes=...`` request header.

    Returns an inclusive ``(start, end)`` pair, or ``None`` when the
    header is malformed or unsatisfiable; the caller maps ``None`` to a
    416 response.
    """

    if size <= 0 or not header:
        return None
    scheme, _, spec = header.partition("=")
    if scheme.strip().lower() != "bytes":
        return None
    # Multiple ranges are legal, but a single media element only needs
    # one; replying with 206 avoids building multipart/byteranges.
    spec = spec.split(",", 1)[0].strip()
    start_text, separator, end_text = spec.partition("-")
    if not separator:
        return None
    try:
        if start_text:
            # ``bytes=start-`` runs to EOF when the end is omitted.
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            # ``bytes=-N`` is a suffix range: the final N bytes.
            length = int(end_text)
            start = max(0, size - length)
            end = size - 1
    except ValueError:
        return None
    if start < 0 or start >= size or end < start:
        return None
    # Clamp overshooting ends (``bytes=0-999999``) to the real size.
    return start, min(end, size - 1)


_INDEX_TEMPLATE = """<!doctype html>
<html lang="{{LANG}}" data-theme="{{THEME_ID}}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#070a0e">
<title>Chandelier Preview</title>
<style>
  :root {
    color-scheme: {{COLOR_SCHEME}};
    {{THEME_VARS}}
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    min-height: 100vh; min-height: 100dvh;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI",
      system-ui, Roboto, "Noto Sans SC", sans-serif;
    -webkit-tap-highlight-color: transparent;
    overscroll-behavior: none;
  }
  button { font: inherit; }
  .app {
    min-height: 100vh; min-height: 100dvh;
    width: min(100%, 1100px); margin: 0 auto; padding: 20px;
    display: flex; flex-direction: column; gap: 18px;
  }
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; padding: 2px;
  }
  .brand {
    display: flex; align-items: center; gap: 10px;
    font-size: .7rem; font-weight: 600; letter-spacing: .22em;
    color: var(--muted);
  }
  .topbar-actions { display: flex; align-items: center; gap: 8px; }
  .gallery-link {
    display: grid; place-items: center; width: 34px; height: 34px;
    border-radius: 50%; background: var(--panel);
    border: 1px solid var(--line); color: var(--muted);
    transition: color .18s, border-color .18s;
  }
  .gallery-link:hover { color: var(--text); border-color: var(--accent); }
  .gallery-link .icon { width: 17px; height: 17px; }
  .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--green);
    box-shadow: 0 0 10px var(--green);
  }
  .live {
    font-size: .56rem; font-weight: 700; letter-spacing: .16em;
    color: var(--green); border: 1px solid var(--green);
    padding: 2px 7px; border-radius: 999px;
  }
  .status {
    font-size: .66rem; color: var(--muted); background: var(--panel);
    border: 1px solid var(--line); padding: 6px 12px;
    border-radius: 999px; white-space: nowrap;
  }
  .stage { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  .tabs {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px;
    padding: 4px; background: var(--panel);
    border: 1px solid var(--line); border-radius: 999px;
  }
  .tab {
    display: flex; align-items: center; justify-content: center; gap: 6px;
    padding: 9px 6px; border: 0; border-radius: 999px;
    background: transparent; color: var(--muted);
    font-size: .7rem; font-weight: 600; cursor: pointer;
    touch-action: manipulation; transition: background .18s, color .18s;
  }
  .tab:hover { color: var(--text); }
  .tab.is-active {
    background: var(--accent); color: var(--accent-ink);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, .1);
  }
  .tab .icon { width: 17px; height: 17px; }
  .frame {
    padding: 8px; border-radius: 20px;
    background: var(--panel);
    border: 1px solid var(--line);
    box-shadow: 0 24px 50px rgba(0, 0, 0, .45);
  }
  img {
    display: block; width: 100%; height: auto; aspect-ratio: 4 / 3;
    object-fit: contain; image-rendering: pixelated;
    background: #000; border-radius: 13px;
    touch-action: manipulation;
  }
  .meta { display: flex; justify-content: center; gap: 8px; }
  .chip {
    font-size: .64rem; color: var(--muted); background: var(--panel);
    border: 1px solid var(--line); border-radius: 999px;
    padding: 5px 12px; text-decoration: none;
  }
  a.chip:hover { color: var(--text); border-color: rgba(255, 255, 255, .22); }
  .remote {
    width: min(100%, 360px); margin: 0 auto;
    display: flex; flex-direction: column; gap: 20px;
    padding: 24px 22px 22px; border-radius: 46px;
    background: var(--panel);
    border: 1px solid var(--line);
    box-shadow:
      0 40px 80px rgba(0, 0, 0, .55),
      inset 0 1px 0 rgba(255, 255, 255, .12),
      inset 0 -2px 0 rgba(0, 0, 0, .6);
  }
  .clickpad {
    position: relative; width: 100%; aspect-ratio: 1; border-radius: 50%;
    background: var(--panel-2);
    border: 1px solid var(--line);
    box-shadow:
      inset 0 2px 1px rgba(255, 255, 255, .1),
      inset 0 -16px 34px rgba(0, 0, 0, .6),
      0 12px 26px rgba(0, 0, 0, .4);
  }
  .clickpad::after {
    content: ""; position: absolute; inset: 11%; border-radius: 50%;
    border: 1px solid rgba(255, 255, 255, .05);
    pointer-events: none;
  }
  .dir, .ok {
    position: absolute; display: grid; place-items: center;
    border: 0; background: transparent; color: var(--muted);
    cursor: pointer; touch-action: manipulation;
    transition: background .15s, color .15s, transform .1s;
  }
  .dir { width: 25%; height: 25%; border-radius: 50%; }
  .dir:hover { color: var(--text); background: rgba(128, 128, 128, .18); }
  .dir:active { background: rgba(128, 128, 128, .3); }
  .dir .icon { width: 22px; height: 22px; }
  .dir.up { top: 1%; left: 50%; transform: translateX(-50%); }
  .dir.down {
    bottom: 1%; left: 50%;
    transform: translateX(-50%) rotate(180deg);
  }
  .dir.left {
    left: 1%; top: 50%;
    transform: translateY(-50%) rotate(-90deg);
  }
  .dir.right {
    right: 1%; top: 50%;
    transform: translateY(-50%) rotate(90deg);
  }
  .ok {
    top: 50%; left: 50%; transform: translate(-50%, -50%);
    width: 38%; aspect-ratio: 1; border-radius: 50%;
    background: var(--accent);
    border: 1px solid var(--accent);
    color: var(--accent-ink);
    font-size: .78rem; font-weight: 700; letter-spacing: .14em;
    box-shadow:
      0 2px 6px rgba(0, 0, 0, .45),
      inset 0 1px 0 rgba(255, 255, 255, .35);
  }
  .ok:hover { filter: brightness(1.08); }
  .ok:active { filter: brightness(.92); }
  .keys { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .key {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    height: 54px; border-radius: 17px;
    border: 1px solid var(--line);
    background: var(--panel-2);
    color: var(--text); cursor: pointer; touch-action: manipulation;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, .1);
    transition: filter .15s, transform .1s;
  }
  .key:hover { filter: brightness(1.15); }
  .key:active { transform: scale(.97); filter: brightness(.92); }
  .key .icon { width: 20px; height: 20px; }
  .key-label { font-size: .74rem; font-weight: 600; letter-spacing: .02em; }
  .icon {
    fill: none; stroke: currentColor; stroke-width: 2;
    stroke-linecap: round; stroke-linejoin: round;
  }
  @media (max-width: 600px) and (orientation: portrait) {
    .app {
      height: 100vh; height: 100dvh; min-height: 0;
      padding: 10px 12px; gap: 8px; overflow: hidden;
    }
    .topbar { padding: 0 2px; }
    .brand { font-size: .64rem; letter-spacing: .18em; }
    .gallery-link { width: 30px; height: 30px; }
    .gallery-link .icon { width: 15px; height: 15px; }
    .status { font-size: .6rem; padding: 4px 9px; }
    .tabs { width: 100%; padding: 3px; gap: 3px; }
    .tab { padding: 6px 4px; font-size: .62rem; gap: 4px; }
    .tab .icon { width: 15px; height: 15px; }
    .stage { flex: 0 0 auto; gap: 6px; align-items: center; }
    .frame {
      width: 100%; padding: 5px;
      border-radius: 16px;
    }
    img { border-radius: 11px; }
    .meta { display: none; }
    .remote {
      flex: none; width: min(100%, 240px); margin: 0 auto;
      padding: 12px 12px 11px; gap: 8px; border-radius: 28px;
    }
    .dir .icon { width: 18px; height: 18px; }
    .ok { font-size: .68rem; letter-spacing: .12em; }
    .keys { gap: 8px; }
    .key { height: 36px; border-radius: 12px; gap: 6px; }
    .key .icon { width: 15px; height: 15px; }
    .key-label { font-size: .64rem; }
  }
  @media (max-width: 380px) {
    .tab span { display: none; }
    .app { padding: 8px 10px; }
  }
  @media (max-width: 600px) and (orientation: portrait) and
    (max-height: 620px) {
    .brand .live { display: none; }
    .remote {
      width: min(100%, 205px);
      padding: 8px 8px 7px; gap: 6px; border-radius: 24px;
    }
    .keys { gap: 6px; }
    .key { height: 32px; border-radius: 10px; }
    .key .icon { width: 13px; height: 13px; }
    .key-label { font-size: .6rem; }
    .tab { padding: 5px 3px; }
  }
  @media (max-width: 600px) and (orientation: portrait) and
    (max-height: 580px) {
    .app { gap: 6px; padding: 6px 8px; }
    .remote {
      width: min(100%, 185px);
      padding: 6px 6px 5px; gap: 5px; border-radius: 22px;
    }
    .key { height: 29px; }
  }
  @media (orientation: landscape) and (max-height: 620px) {
    .app { gap: 10px; padding: 12px 20px; }
    .remote {
      width: min(56vh, 300px); padding: 14px 16px;
      gap: 12px; border-radius: 34px;
    }
    .key { height: 44px; border-radius: 14px; }
    .frame { padding: 6px; }
  }
  @media (min-width: 820px) and (orientation: landscape) {
    .app {
      display: grid; grid-template-columns: minmax(0, 1fr) 340px;
      grid-template-rows: auto minmax(0, 1fr);
      align-items: center; gap: 16px 32px; padding: 24px 32px;
    }
    .topbar { grid-column: 1 / -1; }
    .stage { grid-column: 1; grid-row: 2; }
    .remote {
      grid-column: 2; grid-row: 2;
      width: 100%; max-width: 340px; justify-self: center;
    }
    .frame { padding: 10px; border-radius: 24px; }
  }
  .ambient {
    display: none; position: fixed; inset: -12%; z-index: 0;
    pointer-events: none; overflow: hidden;
  }
  .ambient img {
    width: 100%; height: 100%; object-fit: cover;
    filter: blur(70px) saturate(160%) brightness(.55);
    transform: scale(1.15); opacity: .6;
  }
  .app { position: relative; z-index: 1; }
  html[data-theme="liquid_glass"] .ambient { display: block; }
  html[data-theme="liquid_glass"] body { background: #0b0b0d; }
  html[data-theme="liquid_glass"] .status,
  html[data-theme="liquid_glass"] .tabs,
  html[data-theme="liquid_glass"] .remote,
  html[data-theme="liquid_glass"] .chip,
  html[data-theme="liquid_glass"] .key {
    background: rgba(255, 255, 255, .10);
    border-color: rgba(255, 255, 255, .22);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    backdrop-filter: blur(24px) saturate(180%);
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, .35),
      0 18px 40px rgba(0, 0, 0, .35);
  }
  html[data-theme="liquid_glass"] .remote {
    background: rgba(255, 255, 255, .12);
    border-radius: 40px;
  }
  html[data-theme="liquid_glass"] .clickpad {
    background: rgba(255, 255, 255, .07);
    border-color: rgba(255, 255, 255, .18);
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, .35),
      inset 0 0 44px rgba(255, 255, 255, .06),
      0 8px 24px rgba(0, 0, 0, .25);
  }
  html[data-theme="liquid_glass"] .clickpad::after {
    border-color: rgba(255, 255, 255, .12);
  }
  html[data-theme="liquid_glass"] .ok {
    background: rgba(255, 255, 255, .88);
    border-color: rgba(255, 255, 255, .72);
    box-shadow:
      inset 0 1px 0 #fff,
      0 4px 18px rgba(0, 0, 0, .3);
  }
  html[data-theme="liquid_glass"] .ok:hover { background: #fff; }
  html[data-theme="liquid_glass"] .ok:active {
    background: rgba(255, 255, 255, .72);
  }
  html[data-theme="liquid_glass"] .tab.is-active {
    background: rgba(255, 255, 255, .22);
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, .45),
      0 2px 10px rgba(0, 0, 0, .2);
  }
  html[data-theme="liquid_glass"] .key:hover {
    background: rgba(255, 255, 255, .18);
  }
  html[data-theme="liquid_glass"] .key:active {
    background: rgba(255, 255, 255, .24);
  }
  html[data-theme="liquid_glass"] .frame {
    background: rgba(255, 255, 255, .08);
    border-color: rgba(255, 255, 255, .2);
    -webkit-backdrop-filter: blur(20px) saturate(160%);
    backdrop-filter: blur(20px) saturate(160%);
  }
  html[data-theme="liquid_glass"] .brand { color: rgba(255, 255, 255, .72); }
</style>
</head>
<body>
<div class="ambient" aria-hidden="true"><img id="ambient" alt=""></div>
<main class="app">
  <header class="topbar">
    <div class="brand"><span class="dot"></span>CHANDELIER
      <span class="live">LIVE</span></div>
    <div class="topbar-actions">
      <a class="gallery-link" href="/gallery" aria-label="{{GALLERY_PREVIEW}}" title="{{GALLERY_PREVIEW}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="8.5" cy="9.5" r="1.4"/><path d="M21 15l-4.5-4.5L9 18"/></svg></a>
      <div class="status" id="status">{{CONNECTING}}</div>
    </div>
  </header>
  <section class="stage">
    <nav class="tabs" aria-label="Screens">
      <button class="tab is-active" data-screen="camera" aria-label="{{CAMERA}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h3l2-2h6l2 2h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1z"/><circle cx="12" cy="13" r="3.4"/></svg><span>{{CAMERA}}</span></button>
      <button class="tab" data-screen="gallery" aria-label="{{GALLERY}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="8.5" cy="9.5" r="1.4"/><path d="M21 15l-4.5-4.5L9 18"/></svg><span>{{GALLERY}}</span></button>
      <button class="tab" data-screen="menu" aria-label="{{MENU}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg><span>{{MENU}}</span></button>
      <button class="tab" data-screen="system_monitor" aria-label="{{SYSTEM}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M9 20h6M12 16v4"/></svg><span>{{SYSTEM}}</span></button>
    </nav>
    <div class="frame"><img id="live" alt="{{LIVE_PREVIEW}}" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"></div>
    <div class="meta">
      <span class="chip">320 &times; 240</span>
      <a class="chip" href="/gallery">{{GALLERY_PREVIEW}}</a>
      <a class="chip" href="/snapshot.jpg">{{SNAPSHOT}}</a>
    </div>
  </section>
  <section class="remote" aria-label="Controls">
    <div class="clickpad">
      <button class="dir up" data-action="up" aria-label="{{UP}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5"/><path d="M6 11l6-6 6 6"/></svg></button>
      <button class="dir right" data-action="right" aria-label="{{RIGHT}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5"/><path d="M6 11l6-6 6 6"/></svg></button>
      <button class="dir down" data-action="down" aria-label="{{DOWN}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5"/><path d="M6 11l6-6 6 6"/></svg></button>
      <button class="dir left" data-action="left" aria-label="{{LEFT}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5"/><path d="M6 11l6-6 6 6"/></svg></button>
      <button class="ok" data-action="center" aria-label="{{OK}}">{{OK}}</button>
    </div>
    <div class="keys">
      <button class="key" data-action="cross" aria-label="{{BACK}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 14l-4-4 4-4"/><path d="M5 10h7a4 4 0 0 1 4 4v2"/></svg><span class="key-label">{{BACK}}</span></button>
      <button class="key" data-action="shutter" aria-label="{{SHOT}}"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.5"/></svg><span class="key-label">{{SHOT}}</span></button>
    </div>
  </section>
</main>
<script>
const post = (path, body) => fetch(path, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(body),
}).catch(() => {});

const root = document.documentElement;
const paletteVars = {
  bg: '--bg', text: '--text', muted: '--muted', line: '--line',
  panel: '--panel', panel2: '--panel-2', accent: '--accent',
  accentInk: '--accent-ink', green: '--green',
};
const applyTheme = (theme) => {
  if (!theme || !theme.id) return;
  root.dataset.theme = theme.id;
  if (theme.scheme) root.style.colorScheme = theme.scheme;
  const palette = theme.palette || {};
  Object.entries(paletteVars).forEach(([key, cssVar]) => {
    if (palette[key]) root.style.setProperty(cssVar, palette[key]);
  });
};
// Poll while visible so a theme change made on the device repaints this
// page too; hidden tabs skip the request.
const refreshTheme = () => {
  if (document.hidden) return;
  fetch('/status', {cache: 'no-store'})
    .then((response) => response.ok ? response.json() : null)
    .then((data) => { if (data) applyTheme(data.theme); })
    .catch(() => {});
};
setInterval(refreshTheme, 3000);

document.querySelectorAll('[data-action]').forEach((button) => {
  button.addEventListener('click', () => {
    post('/control', {action: button.dataset.action});
  });
});
document.querySelectorAll('[data-screen]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-screen]').forEach((item) => {
      item.classList.toggle('is-active', item === button);
    });
    post('/screen', {id: button.dataset.screen});
  });
});

const live = document.getElementById('live');
const ambient = document.getElementById('ambient');
const status = document.getElementById('status');
// Send taps as 0..1 fractions so the device can scale them to its LCD.
live.addEventListener('click', (event) => {
  const rect = live.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const x = (event.clientX - rect.left) / rect.width;
  const y = (event.clientY - rect.top) / rect.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  post('/touch', {x: x, y: y});
});
// EventSource delivers one JPEG per event and reconnects on its own;
// browsers without it fall back to the MJPEG stream below.
if (window.EventSource) {
  const source = new EventSource('/events');
  source.onopen = () => { status.textContent = {{STREAM_CONNECTED}}; };
  source.onerror = () => { status.textContent = {{RECONNECTING}}; };
  let frames = 0;
  source.onmessage = (event) => {
    const frame = 'data:image/jpeg;base64,' + event.data;
    live.src = frame;
    // The ambient backdrop is heavily blurred; update it less often.
    if (++frames % 5 === 0) ambient.src = frame;
  };
} else {
  live.src = '/stream.mjpg';
  ambient.src = '/snapshot.jpg';
  status.textContent = {{MJPEG}};
}
</script>
</body>
</html>
"""

_GALLERY_TEMPLATE = """<!doctype html>
<html lang="{{LANG}}" data-theme="{{THEME_ID}}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#070a0e">
<title>{{GALLERY}} · Chandelier</title>
<style>
  :root {
    color-scheme: {{COLOR_SCHEME}};
    {{THEME_VARS}}
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    min-height: 100vh; min-height: 100dvh;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI",
      system-ui, Roboto, "Noto Sans SC", sans-serif;
    -webkit-tap-highlight-color: transparent;
  }
  body.viewing { overflow: hidden; }
  a { color: inherit; text-decoration: none; }
  button { font: inherit; }
  .app {
    width: min(100%, 1100px); margin: 0 auto; padding: 18px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px;
  }
  .pill {
    display: inline-flex; align-items: center; justify-content: center; gap: 7px;
    height: 36px; padding: 0 14px;
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 999px; color: var(--muted);
    font-size: .72rem; font-weight: 600; cursor: pointer;
    transition: color .18s, border-color .18s, background .18s;
  }
  .pill:not(:disabled):hover {
    color: var(--text); border-color: var(--accent);
  }
  .pill:disabled { opacity: .45; cursor: default; }
  .pill.is-active {
    color: var(--accent-ink); background: var(--accent);
    border-color: var(--accent);
  }
  .pill.danger { color: #ff6b6b; border-color: rgba(255, 107, 107, .55); }
  .pill.danger:not(:disabled):hover {
    color: #fff; background: #d94b4b; border-color: #d94b4b;
  }
  .topbar-actions { display: flex; align-items: center; gap: 8px; }
  .icon {
    width: 16px; height: 16px; fill: none; stroke: currentColor;
    stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;
  }
  .title {
    display: flex; align-items: baseline; gap: 8px;
    font-size: .86rem; font-weight: 700; letter-spacing: .08em;
  }
  .count {
    color: var(--muted); font-size: .66rem; font-weight: 500;
    letter-spacing: 0; font-variant-numeric: tabular-nums;
  }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 6px;
  }
  .tile {
    position: relative; display: block; width: 100%; aspect-ratio: 1;
    padding: 0; overflow: hidden; cursor: pointer;
    border: 1px solid var(--line); border-radius: 14px;
    background: var(--panel);
    transition: transform .16s, border-color .16s;
  }
  .tile:hover { transform: translateY(-1px); border-color: var(--accent); }
  .tile:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .tile .check {
    position: absolute; top: 6px; right: 6px;
    display: none; place-items: center;
    width: 20px; height: 20px; border-radius: 50%;
    background: var(--accent); color: var(--accent-ink);
    font-size: .68rem; font-weight: 700; line-height: 1;
  }
  .tile.is-selected {
    border-color: var(--accent);
    box-shadow: inset 0 0 0 2px var(--accent);
  }
  .tile.is-selected .check { display: grid; }
  .tile img {
    display: block; width: 100%; height: 100%;
    object-fit: cover; background: var(--panel-2);
  }
  .tile.is-video {
    display: grid; place-items: center;
    background: radial-gradient(circle at 30% 20%, var(--panel-2), var(--panel));
  }
  .play {
    width: 0; height: 0;
    border-top: 15px solid transparent;
    border-bottom: 15px solid transparent;
    border-left: 24px solid var(--muted);
    filter: drop-shadow(0 2px 8px rgba(0, 0, 0, .5));
  }
  .empty {
    padding: 72px 20px; text-align: center; color: var(--muted);
    font-size: .8rem;
  }
  .empty[hidden] { display: none; }
  .manage-bar {
    position: fixed; left: 50%; bottom: 12px; transform: translateX(-50%);
    z-index: 30; display: flex; align-items: center; gap: 8px;
    padding: 7px 9px; border-radius: 999px;
    background: var(--panel); border: 1px solid var(--line);
    box-shadow: 0 16px 36px rgba(0, 0, 0, .45);
    -webkit-backdrop-filter: blur(20px); backdrop-filter: blur(20px);
    max-width: calc(100vw - 16px);
  }
  .manage-bar[hidden] { display: none; }
  .manage-bar .count {
    font-size: .7rem; color: var(--muted); padding: 0 4px;
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  body.managing .app { padding-bottom: 76px; }
  .toast {
    position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
    z-index: 40; padding: 8px 14px; border-radius: 999px;
    background: var(--panel); border: 1px solid var(--line);
    color: var(--text); font-size: .7rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, .35);
  }
  .toast[hidden] { display: none; }
  .viewer {
    position: fixed; inset: 0; z-index: 50;
    display: flex; align-items: center; justify-content: center;
    padding: 16px; background: rgba(0, 0, 0, .9);
  }
  .viewer[hidden] { display: none; }
  .viewer figure {
    margin: 0; max-width: 100%; max-height: 100%;
    display: flex; flex-direction: column; align-items: center; gap: 10px;
  }
  .viewer img, .viewer video {
    max-width: calc(100vw - 32px); max-height: calc(100vh - 120px);
    max-height: calc(100dvh - 120px);
    object-fit: contain; border-radius: 10px; background: #000;
  }
  .caption {
    max-width: 90vw; color: rgba(255, 255, 255, .72);
    font-size: .68rem; text-align: center; word-break: break-all;
  }
  .viewer .close {
    position: fixed; top: 14px; right: 14px;
    width: 40px; height: 40px; border-radius: 50%;
    border: 1px solid rgba(255, 255, 255, .2);
    background: rgba(20, 20, 22, .7); color: #fff;
    font-size: 1.35rem; line-height: 1; cursor: pointer;
    -webkit-backdrop-filter: blur(16px); backdrop-filter: blur(16px);
  }
  .nav {
    position: fixed; top: 50%; transform: translateY(-50%);
    width: 44px; height: 60px; border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, .16);
    background: rgba(20, 20, 22, .55); color: #fff;
    font-size: 1.6rem; line-height: 1; cursor: pointer;
    -webkit-backdrop-filter: blur(16px); backdrop-filter: blur(16px);
  }
  .nav.prev { left: 10px; }
  .nav.next { right: 10px; }
  .nav[disabled] { opacity: .25; cursor: default; }
  .download {
    flex: none; padding: 7px 14px; border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, .2);
    background: rgba(255, 255, 255, .1); color: #fff;
    font-size: .68rem; font-weight: 600;
  }
  button.download { font: inherit; cursor: pointer; }
  .download.danger {
    color: #ffb3b3; border-color: rgba(255, 107, 107, .6);
  }
  .viewer-actions { display: flex; align-items: center; gap: 8px; }
  html[data-theme="liquid_glass"] .pill {
    background: rgba(255, 255, 255, .10);
    border-color: rgba(255, 255, 255, .22);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    backdrop-filter: blur(24px) saturate(180%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, .35);
  }
  html[data-theme="liquid_glass"] .tile {
    border-color: rgba(255, 255, 255, .2);
  }
  @media (min-width: 700px) {
    .grid { grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; }
  }
  @media (max-width: 480px) {
    .app { padding: 12px; gap: 12px; }
    .pill { height: 32px; padding: 0 11px; font-size: .66rem; }
    .grid { grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); gap: 4px; }
    .tile { border-radius: 11px; }
    .nav { width: 36px; height: 50px; }
    .manage-bar {
      bottom: 8px; gap: 5px; padding: 6px;
      flex-wrap: wrap; justify-content: center; border-radius: 18px;
    }
    .manage-bar .pill { height: 30px; padding: 0 9px; }
    body.managing .app { padding-bottom: 66px; }
  }
  @media (max-width: 400px) {
    .topbar-actions .pill span { display: none; }
    .topbar-actions .pill { padding: 0 10px; }
  }
</style>
</head>
<body>
<main class="app">
  <header class="topbar">
    <a class="pill" href="/" aria-label="{{BACK}}">
      <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7"/></svg>
      <span>{{BACK}}</span>
    </a>
    <div class="title">{{GALLERY}} <span class="count" id="count">0</span></div>
    <div class="topbar-actions">
      <button class="pill" id="refresh" type="button" aria-label="{{REFRESH}}">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2.3 6.3"/><path d="M20 4v7h-7"/></svg>
        <span>{{REFRESH}}</span>
      </button>
      <button class="pill" id="manage" type="button" aria-label="{{MANAGE}}">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20h4L18 10l-4-4L4 16v4z"/><path d="M14 6l4 4"/></svg>
        <span>{{MANAGE}}</span>
      </button>
    </div>
  </header>
  <div class="grid" id="grid"></div>
  <div class="empty" id="empty" hidden>{{NO_MEDIA}}</div>
</main>

<div class="manage-bar" id="manage-bar" hidden>
  <span class="count">{{SELECTED}} <span id="selected-count">0</span></span>
  <button class="pill" id="select-all" type="button">{{SELECT_ALL}}</button>
  <button class="pill danger" id="delete-selected" type="button">{{DELETE}}</button>
  <button class="pill" id="done" type="button">{{DONE}}</button>
</div>
<div class="toast" id="toast" hidden></div>

<div class="viewer" id="viewer" role="dialog" aria-modal="true" aria-label="{{GALLERY}}" hidden>
  <button class="nav prev" id="prev" type="button" aria-label="{{PREVIOUS}}">‹</button>
  <figure>
    <img id="viewer-image" alt="" aria-describedby="caption" hidden>
    <video id="viewer-video" controls playsinline preload="metadata" hidden></video>
    <figcaption class="caption" id="caption"></figcaption>
    <div class="viewer-actions">
      <a class="download" id="download" href="#" download>{{DOWNLOAD}}</a>
      <button class="download danger" id="delete-current" type="button">{{DELETE}}</button>
    </div>
  </figure>
  <button class="nav next" id="next" type="button" aria-label="{{NEXT}}">›</button>
  <button class="close" id="close" type="button" aria-label="{{CLOSE}}">×</button>
</div>

<script>
const grid = document.getElementById('grid');
const empty = document.getElementById('empty');
const count = document.getElementById('count');
const viewer = document.getElementById('viewer');
const viewerImage = document.getElementById('viewer-image');
const viewerVideo = document.getElementById('viewer-video');
const caption = document.getElementById('caption');
const download = document.getElementById('download');
const previousButton = document.getElementById('prev');
const nextButton = document.getElementById('next');
const refreshButton = document.getElementById('refresh');
const closeButton = document.getElementById('close');
const manageButton = document.getElementById('manage');
const manageBar = document.getElementById('manage-bar');
const selectedCount = document.getElementById('selected-count');
const selectAllButton = document.getElementById('select-all');
const deleteSelectedButton = document.getElementById('delete-selected');
const doneButton = document.getElementById('done');
const deleteCurrentButton = document.getElementById('delete-current');
const toast = document.getElementById('toast');

let items = [];
let current = -1;
let currentItemId = null;
let signature = null;
let lastFocused = null;
let manageMode = false;
const selected = new Set();
let toastTimer = null;

// Every media URL carries the item version so a replacement that keeps
// the same filename is never served from the browser cache.
const mediaUrl = (item) =>
  '/media/' + encodeURIComponent(item.id) + '?v=' + item.version;
const viewUrl = (item) =>
  '/view/' + encodeURIComponent(item.id) + '?v=' + item.version;
const thumbUrl = (item) =>
  '/thumb/' + encodeURIComponent(item.id) +
  '?w=400&h=400&v=' + item.version;

const formatSize = (bytes) => {
  const value = Number(bytes) || 0;
  if (value >= 1e9) return (value / 1e9).toFixed(1) + ' GB';
  if (value >= 1e6) return (value / 1e6).toFixed(1) + ' MB';
  if (value >= 1e3) return Math.round(value / 1e3) + ' KB';
  return value + ' B';
};
const formatDate = (seconds) => {
  const value = Number(seconds) || 0;
  return value ? new Date(value * 1000).toLocaleString() : '';
};

// Reload the listing; the signature check below skips re-rendering when
// an auto-refresh found no changed IDs or versions, so the grid does not
// flicker or lose the user's place every 15 s.
async function load(force) {
  let payload;
  try {
    const response = await fetch(
      '/api/gallery' + (force ? '?refresh=1' : ''),
      {cache: 'no-store'}
    );
    payload = await response.json();
  } catch (error) {
    // Keep the last good listing on a transient network error.
    return;
  }
  items = Array.isArray(payload.items) ? payload.items : [];
  if (manageMode) {
    const valid = new Set(items.map((item) => item.id));
    for (const itemId of Array.from(selected)) {
      if (!valid.has(itemId)) selected.delete(itemId);
    }
    if (!items.length) {
      manageMode = false;
      selected.clear();
    }
  }
  const nextSignature = items
    .map((item) => item.id + ':' + item.version)
    .join('|');
  if (!force && nextSignature === signature) return;
  signature = nextSignature;
  if (!viewer.hidden && currentItemId) {
    const index = items.findIndex((item) => item.id === currentItemId);
    if (index < 0) {
      closeViewer();
    } else {
      current = index;
    }
  }
  render();
}

function render() {
  grid.replaceChildren();
  count.textContent = String(items.length);
  empty.hidden = items.length > 0;
  const fragment = document.createDocumentFragment();
  items.forEach((item, index) => {
    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = 'tile' + (item.type === 'video' ? ' is-video' : '');
    tile.setAttribute('aria-label', item.name);
    tile.addEventListener('click', () => {
      if (manageMode) {
        toggleSelection(item.id, tile);
      } else {
        openViewer(index);
      }
    });
    if (item.type === 'video') {
      const play = document.createElement('span');
      play.className = 'play';
      tile.append(play);
    } else {
      const image = document.createElement('img');
      image.loading = 'lazy';
      image.decoding = 'async';
      image.alt = '';
      image.addEventListener('error', () => image.remove());
      image.src = thumbUrl(item);
      tile.append(image);
    }
    const check = document.createElement('span');
    check.className = 'check';
    check.textContent = '✓';
    tile.append(check);
    if (manageMode && selected.has(item.id)) {
      tile.classList.add('is-selected');
    }
    fragment.append(tile);
  });
  grid.append(fragment);
  updateManageUi();
}

// The viewer is a modal dialog: remember the focused element so closing
// can restore it, and mark the page behind it inert for keyboard and
// screen-reader users.
function openViewer(index) {
  if (!items.length) return;
  current = ((index % items.length) + items.length) % items.length;
  const item = items[current];
  const isVideo = item.type === 'video';
  const url = mediaUrl(item);
  const wasHidden = viewer.hidden;
  if (wasHidden) lastFocused = document.activeElement;

  viewerImage.hidden = isVideo;
  viewerVideo.hidden = !isVideo;
  if (isVideo) {
    viewerVideo.src = url;
    viewerVideo.play().catch(() => {});
  } else {
    // Stepping from a video to a photo must stop the hidden player.
    viewerVideo.pause();
    viewerVideo.removeAttribute('src');
    viewerVideo.load();
    viewerImage.alt = item.name;
    viewerImage.src = viewUrl(item);
  }
  caption.textContent = [
    item.name, formatSize(item.size), formatDate(item.mtime),
  ].filter(Boolean).join(' · ');
  download.href = url;
  download.setAttribute('download', item.name);
  currentItemId = item.id;
  viewer.hidden = false;
  document.body.classList.add('viewing');
  const app = document.querySelector('.app');
  if (app) app.inert = true;
  if (wasHidden) closeButton.focus();

  const single = items.length < 2;
  previousButton.disabled = single;
  nextButton.disabled = single;
}

function closeViewer() {
  viewer.hidden = true;
  // Clear the source as well as pausing: otherwise the browser keeps
  // the decoder and the file handle alive while the dialog is closed.
  viewerVideo.pause();
  viewerVideo.removeAttribute('src');
  viewerVideo.load();
  viewerImage.removeAttribute('src');
  viewerImage.alt = '';
  document.body.classList.remove('viewing');
  currentItemId = null;
  const app = document.querySelector('.app');
  if (app) app.inert = false;
  if (lastFocused && lastFocused.focus) lastFocused.focus();
  lastFocused = null;
}

function step(delta) {
  if (items.length < 2) return;
  openViewer(current + delta);
}

// Manage mode keeps a set of item IDs, not DOM nodes, so the 15 s refresh
// can rebuild the grid without losing the selection.
function updateManageUi() {
  manageButton.classList.toggle('is-active', manageMode);
  manageBar.hidden = !manageMode;
  document.body.classList.toggle('managing', manageMode);
  selectedCount.textContent = String(selected.size);
  deleteSelectedButton.disabled = selected.size === 0;
  const allSelected = items.length > 0 && selected.size === items.length;
  selectAllButton.textContent = allSelected
    ? {{CLEAR_JS}}
    : {{SELECT_ALL_JS}};
}

function toggleSelection(itemId, tile) {
  if (selected.has(itemId)) {
    selected.delete(itemId);
    tile.classList.remove('is-selected');
  } else {
    selected.add(itemId);
    tile.classList.add('is-selected');
  }
  updateManageUi();
}

// ---- Manage mode ----
function enterManage() {
  if (!items.length) return;
  manageMode = true;
  closeViewer();
  render();
}

function exitManage() {
  manageMode = false;
  selected.clear();
  render();
}

function selectAll() {
  if (selected.size === items.length) {
    selected.clear();
  } else {
    items.forEach((item) => selected.add(item.id));
  }
  render();
}

function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.hidden = true;
  }, 2500);
}

// The server deletes the file plus its on-device render cache; the response
// separates successful and failed IDs so the UI can keep retryable items.
async function deleteItems(ids) {
  const response = await fetch('/api/gallery/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ids: ids}),
  });
  if (!response.ok) throw new Error('delete failed');
  return response.json();
}

async function deleteSelected() {
  const ids = Array.from(selected);
  if (!ids.length) return;
  if (!confirm({{DELETE_CONFIRM}} + ' (' + ids.length + ')')) return;

  deleteSelectedButton.disabled = true;
  try {
    const payload = await deleteItems(ids);
    const count = (payload.deleted || []).length;
    const failed = payload.failed || [];
    selected.clear();
    failed.forEach((itemId) => selected.add(itemId));
    showToast(
      failed.length
        ? {{DELETE_FAILED}} + ' ' + failed.length
        : {{DELETED}} + ' ' + count
    );
    await load(true);
  } catch (error) {
    showToast({{DELETE_FAILED}});
  } finally {
    updateManageUi();
  }
}

async function deleteCurrent() {
  if (!currentItemId) return;
  if (!confirm({{DELETE_CONFIRM}} + ' (1)')) return;

  try {
    const payload = await deleteItems([currentItemId]);
    const count = (payload.deleted || []).length;
    // Close before reloading: the item the viewer points at is gone.
    closeViewer();
    showToast({{DELETED}} + ' ' + count);
    await load(true);
  } catch (error) {
    showToast({{DELETE_FAILED}});
  }
}

previousButton.addEventListener('click', () => step(-1));
nextButton.addEventListener('click', () => step(1));
closeButton.addEventListener('click', closeViewer);
refreshButton.addEventListener('click', () => load(true));
manageButton.addEventListener('click', () => {
  if (manageMode) {
    exitManage();
  } else {
    enterManage();
  }
});
doneButton.addEventListener('click', exitManage);
selectAllButton.addEventListener('click', selectAll);
deleteSelectedButton.addEventListener('click', deleteSelected);
deleteCurrentButton.addEventListener('click', deleteCurrent);
viewer.addEventListener('click', (event) => {
  if (event.target === viewer) closeViewer();
});

// Modal keyboard handling: Escape closes, Tab is trapped inside the
// open dialog, and arrows step between items unless a video control or
// form element has focus.
document.addEventListener('keydown', (event) => {
  if (viewer.hidden) {
    if (manageMode && event.key === 'Escape') {
      event.preventDefault();
      exitManage();
    }
    return;
  }
  if (event.key === 'Escape') {
    event.preventDefault();
    closeViewer();
    return;
  }
  if (event.key === 'Tab') {
    const focusable = Array.from(
      viewer.querySelectorAll('button, a[href], video[controls]')
    ).filter((element) => !element.disabled && !element.hidden);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
    return;
  }
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
  const target = event.target;
  if (
    target instanceof Element &&
    target.closest('video, input, textarea, select, [contenteditable]')
  ) {
    // Leave keyboard seeking and video controls alone.
    return;
  }
  event.preventDefault();
  step(event.key === 'ArrowLeft' ? -1 : 1);
});

// Swipe navigation: a deliberate horizontal drag steps between items.
// The 60 px threshold keeps taps and vertical gestures from navigating.
let touchStartX = null;
viewer.addEventListener('touchstart', (event) => {
  const target = event.target;
  if (
    target instanceof Element &&
    target.closest('video, a, button, input')
  ) {
    // Never turn a scrub or a button tap into a page swipe.
    touchStartX = null;
    return;
  }
  touchStartX = event.changedTouches[0].clientX;
}, {passive: true});
viewer.addEventListener('touchend', (event) => {
  if (touchStartX === null) return;
  const delta = event.changedTouches[0].clientX - touchStartX;
  touchStartX = null;
  if (Math.abs(delta) > 60) step(delta > 0 ? -1 : 1);
}, {passive: true});

const paletteVars = {
  bg: '--bg', text: '--text', muted: '--muted', line: '--line',
  panel: '--panel', panel2: '--panel-2', accent: '--accent',
  accentInk: '--accent-ink', green: '--green',
};
const applyTheme = (theme) => {
  if (!theme || !theme.id) return;
  const root = document.documentElement;
  root.dataset.theme = theme.id;
  if (theme.scheme) root.style.colorScheme = theme.scheme;
  const palette = theme.palette || {};
  Object.entries(paletteVars).forEach(([key, cssVar]) => {
    if (palette[key]) root.style.setProperty(cssVar, palette[key]);
  });
};
// Poll while visible so a theme change made on the device repaints this
// page too; hidden tabs skip the request.
const refreshTheme = () => {
  if (document.hidden) return;
  fetch('/status', {cache: 'no-store'})
    .then((response) => response.ok ? response.json() : null)
    .then((data) => { if (data) applyTheme(data.theme); })
    .catch(() => {});
};
setInterval(refreshTheme, 5000);
// Catch up on listing changes missed while the tab was hidden.
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && viewer.hidden) load(false);
});
// Pick up photos captured or copied onto the device while the page stays
// open, but never reorder the grid under an open viewer.
setInterval(() => {
  if (viewer.hidden) load(false);
}, 15000);

load(true);
</script>
</body>
</html>
"""


def _css_color(color) -> str:
    """Convert a BGR theme colour into a CSS ``rgb()`` string."""

    blue, green, red = (int(value) for value in color)
    return f"rgb({red}, {green}, {blue})"


def _theme_palette(theme) -> dict[str, str]:
    """Map a theme's BGR colours onto CSS ``rgb()`` strings."""

    return {
        "bg": _css_color(theme.background),
        "text": _css_color(theme.text),
        "muted": _css_color(theme.text_dim),
        "line": _css_color(theme.border),
        "panel": _css_color(theme.surface),
        "panel2": _css_color(theme.surface_alt),
        "accent": _css_color(theme.accent),
        "accentInk": _css_color(theme.accent_text),
        "green": _css_color(theme.success),
    }


def _theme_vars(theme) -> str:
    """Render the palette as CSS custom-property declarations."""

    names = {
        "bg": "--bg",
        "text": "--text",
        "muted": "--muted",
        "line": "--line",
        "panel": "--panel",
        "panel2": "--panel-2",
        "accent": "--accent",
        "accentInk": "--accent-ink",
        "green": "--green",
    }
    palette = _theme_palette(theme)
    return " ".join(
        f"{names[key]}: {value};" for key, value in palette.items()
    )


def _theme_payload() -> dict:
    theme = get_theme(active_theme_id())
    return {
        "id": theme.id,
        "scheme": "light" if _is_light_theme(theme) else "dark",
        "palette": _theme_palette(theme),
    }


def _is_light_theme(theme) -> bool:
    """Return whether the theme background is light.

    Uses perceptual luma (ITU-R BT.601) rather than a channel average so
    the browser's ``color-scheme`` matches the device's contrast choice.
    """

    blue, green, red = (int(value) for value in theme.background)
    return 0.299 * red + 0.587 * green + 0.114 * blue >= 140


def _index_html() -> bytes:
    """Render the remote-control page with translated labels."""

    theme = get_theme(active_theme_id())
    values = {
        "LANG": escape(translator.language),
        "LIVE_PREVIEW": escape(tr("Live preview")),
        "THEME_ID": escape(theme.id),
        "THEME_VARS": _theme_vars(theme),
        "COLOR_SCHEME": "light" if _is_light_theme(theme) else "dark",
        "CAMERA": escape(tr("Camera")),
        "GALLERY": escape(tr("Gallery")),
        "GALLERY_PREVIEW": escape(tr("Gallery preview")),
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
    return _fill_template(_INDEX_TEMPLATE, values)


def _gallery_html() -> bytes:
    """Render the gallery page with translated labels."""

    theme = get_theme(active_theme_id())
    values = {
        "LANG": escape(translator.language),
        "THEME_ID": escape(theme.id),
        "THEME_VARS": _theme_vars(theme),
        "COLOR_SCHEME": "light" if _is_light_theme(theme) else "dark",
        "GALLERY": escape(tr("Gallery")),
        "BACK": escape(tr("Back")),
        "REFRESH": escape(tr("Refresh")),
        "NO_MEDIA": escape(tr("No media")),
        "PREVIOUS": escape(tr("Previous")),
        "NEXT": escape(tr("Next")),
        "CLOSE": escape(tr("Close")),
        "DOWNLOAD": escape(tr("Download")),
        "MANAGE": escape(tr("Manage")),
        "DONE": escape(tr("Done")),
        "SELECT_ALL": escape(tr("Select all")),
        "CLEAR": escape(tr("Clear")),
        "SELECT_ALL_JS": json.dumps(tr("Select all")),
        "CLEAR_JS": json.dumps(tr("Clear")),
        "SELECTED": escape(tr("Selected")),
        "DELETE": escape(tr("Delete")),
        "DELETE_CONFIRM": json.dumps(tr("Delete selected items?")),
        "DELETED": json.dumps(tr("Deleted")),
        "DELETE_FAILED": json.dumps(tr("Delete failed")),
    }
    return _fill_template(_GALLERY_TEMPLATE, values)


def _fill_template(template: str, values: dict[str, str]) -> bytes:
    """Substitute ``{{TOKEN}}`` placeholders and encode as UTF-8."""

    for token, value in values.items():
        template = template.replace("{{" + token + "}}", value)
    return template.encode("utf-8")


def load_auth_file(path: str | Path) -> tuple[str, str] | None:
    """Load HTTP Basic credentials from a small JSON file.

    The file is kept separate from ``config.json`` so it can be untracked.
    A missing or invalid file disables authentication, which keeps existing
    installations working until credentials are configured.
    """

    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning(
            "could not read preview auth file %s", path, exc_info=True
        )
        return None

    if not isinstance(raw, dict):
        logger.warning("preview auth file %s must contain a JSON object", path)
        return None

    username = raw.get("username")
    password = raw.get("password")
    if not isinstance(username, str) or not isinstance(password, str):
        logger.warning(
            "preview auth file %s needs string 'username' and 'password'",
            path,
        )
        return None

    username = username.strip()
    if not username or not password:
        logger.warning(
            "preview auth disabled: username and password must be non-empty "
            "in %s",
            path,
        )
        return None
    return username, password


class PreviewServer:
    """Publish composed frames and serve the latest one to browsers.

    The HTTP server runs in its own daemon thread and gives every request
    its own handler thread.  Frames are copied on publish, JPEG-encoded
    once per sequence, and then shared by all connected clients.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        auth: tuple[str, str] | None = None,
        media_path: str | Path | None = None,
    ):
        self._host = host
        self._port = int(port)
        self._auth = self._normalise_auth(auth)
        self._gallery = WebGallery(media_path)
        # Lifecycle fields use their own lock so start()/stop() can block
        # without stalling threads waiting on frame state.
        self._state_lock = threading.Lock()
        # The condition guards the frame, sequence and cached JPEG, and
        # wakes stream handlers whenever publish() advances the sequence.
        self._condition = threading.Condition()
        # Only one thread encodes at a time; every stream then reuses the
        # resulting JPEG instead of running cv2.imencode itself.
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
        self._touch_handler = None
        # Open MJPEG/SSE streams, reported by /status.
        self._clients = 0

    @staticmethod
    def _normalise_auth(
        auth: tuple[str, str] | None,
    ) -> tuple[str, str] | None:
        """Validate credentials, disabling auth on malformed input.

        Bad configuration must not crash the server or lock the user out
        silently, so it falls back to the previous open-LAN behaviour.
        """

        if auth is None:
            return None
        try:
            username, password = auth
        except (TypeError, ValueError):
            logger.warning(
                "preview auth ignored: expected a (username, password) pair"
            )
            return None
        username = str(username)
        password = str(password)
        if not username or not password:
            logger.warning(
                "preview auth ignored: username and password are required"
            )
            return None
        return username, password

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def auth(self) -> tuple[str, str] | None:
        return self._auth

    @property
    def gallery(self) -> WebGallery:
        return self._gallery

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
        """Apply the enabled flag; returns False only if starting failed."""

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

    def set_touch_handler(self, handler) -> None:
        """Register ``handler(x, y) -> bool`` for taps on the preview."""

        self._touch_handler = handler

    def send_touch(self, x: float, y: float) -> bool:
        """Forward normalised (0..1) coordinates to the touch handler."""

        handler = self._touch_handler
        if handler is None:
            return False
        try:
            return bool(handler(float(x), float(y)))
        except Exception:
            logger.exception("preview touch failed")
            return False

    def send_action(self, name: str) -> bool:
        """Deliver a remote button name as an ``Action`` press event."""

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
        """Ask the controller to show ``screen_id``; unknown IDs fail."""

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
        """Number of open MJPEG/SSE streams."""

        with self._condition:
            return self._clients

    def client_open(self) -> None:
        """Track one streaming client for /status."""

        with self._condition:
            self._clients += 1

    def client_close(self) -> None:
        """Release a streaming client; clamped against double-close."""

        with self._condition:
            self._clients = max(0, self._clients - 1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Bind and start the HTTP server; False if the port is taken."""

        with self._state_lock:
            if self._enabled:
                return True
            # Each stop() replaces this event so handlers blocked in an
            # old stream generation wake up instead of waiting forever.
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
                # Drop stale frames from a previous server generation.
                self._frame = None
                self._jpeg = None
                self._jpeg_sequence = -1
                self._clients = 0
            self._server = server
            self._thread = threading.Thread(
                target=server.serve_forever,
                # A short poll interval makes shutdown() responsive.
                kwargs={"poll_interval": 0.1},
                name="preview-server",
                # Daemon so a wedged accept loop cannot block exit.
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
        """Stop the server and wake every blocked stream handler."""

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
            # Wake streams that are waiting for the next frame so they
            # notice the stop event instead of sleeping out their timeout.
            self._condition.notify_all()

        if server is not None:
            # shutdown() ends the accept loop; server_close() frees the
            # port so a quick re-enable can bind again.
            server.shutdown()
            server.server_close()
        if thread is not None:
            # Do not wait forever on a handler stuck writing to a peer.
            thread.join(timeout=2.0)

        with self._condition:
            # Clear the frame and bump the sequence so late clients never
            # receive a picture from the stopped generation.
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
        # Copy straight away: the controller may reuse its frame buffer
        # while HTTP threads are still encoding or sending it.
        snapshot = frame.copy()
        with self._condition:
            # Re-check: stop() may have raced the unlocked check above.
            if not self._enabled:
                return
            self._frame = snapshot
            self._sequence += 1
            self._condition.notify_all()

    def latest_frame(self):
        """Return ``(sequence, frame)`` atomically, or ``None``."""

        with self._condition:
            if self._frame is None:
                return None
            return self._sequence, self._frame

    def jpeg_for(self, sequence: int, frame) -> bytes | None:
        """Encode ``frame`` once per sequence and share it with all clients."""

        # One encode per sequence: N viewers should not trigger N
        # cv2.imencode calls on the Pi.
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
                    _JPEG_PARAMS,
                )
            except cv2.error:
                logger.debug("failed to encode preview frame", exc_info=True)
                return None
            if not ok:
                return None

            jpeg = buffer.tobytes()
            with self._condition:
                # A slower encoder must not replace a newer cached frame.
                if sequence >= self._jpeg_sequence:
                    self._jpeg = jpeg
                    self._jpeg_sequence = sequence
            return jpeg

    def snapshot(self) -> bytes | None:
        """Encode the latest frame for the one-shot ``/snapshot.jpg``."""

        latest = self.latest_frame()
        if latest is None:
            return None
        sequence, frame = latest
        return self.jpeg_for(sequence, frame)

    def wait_for_frame(
        self, after: int, timeout: float = 1.0
    ):
        """Wait for a frame newer than ``after``, or until timeout.

        This performs a single wait rather than a loop: streaming
        handlers re-check the sequence themselves and use the timeout to
        notice peers that have gone away.
        """

        with self._condition:
            if self._sequence == after or self._frame is None:
                self._condition.wait(timeout)
            if self._frame is None:
                return None
            return self._sequence, self._frame


class _PreviewHTTPServer(ThreadingHTTPServer):
    """Thread-per-request HTTP server shared by every preview route."""

    # Streaming handlers live as long as a browser tab; do not join them
    # when the interpreter exits.
    daemon_threads = True
    # The feature is toggled at runtime; rebinding should not wait for
    # the previous socket to leave TIME_WAIT.
    allow_reuse_address = True

    # Installed by PreviewServer.start() before serve_forever() runs.
    preview: PreviewServer


class _PreviewHandler(BaseHTTPRequestHandler):
    """Route one browser request to the preview API.

    The threading server creates one instance per connection, so a slow
    MJPEG or SSE stream never blocks the control and gallery routes.
    """

    # Modern clients expect HTTP/1.1 semantics; every response still
    # sends an explicit Content-Length and closes the socket.
    protocol_version = "HTTP/1.1"
    # Socket timeout so a stalled peer cannot pin a handler thread.
    timeout = 10
    # MJPEG/SSE writes are small and latency-sensitive; Nagle would batch
    # them and add visible delay to the stream.
    disable_nagle_algorithm = True

    def log_message(self, format, *args) -> None:
        logger.debug("preview http: " + format, *args)

    def _authorized(self) -> bool:
        """Check HTTP Basic credentials when any are configured."""

        auth = self.server.preview.auth
        if auth is None:
            # No credentials configured: keep the original open-LAN
            # behaviour rather than locking every client out.
            return True

        header = self.headers.get("Authorization", "")
        scheme, _, value = header.partition(" ")
        value = value.strip()
        if scheme.lower() != "basic" or not value:
            return False

        try:
            # validate=True rejects stray characters instead of silently
            # discarding them, which keeps malformed tokens a 401.
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            return False
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            return False

        username, separator, password = decoded.partition(":")
        if not separator:
            return False

        expected_username, expected_password = auth
        # Constant-time comparisons avoid leaking how much of a guessed
        # username or password was correct through response timing.
        user_ok = hmac.compare_digest(
            username.encode("utf-8"), expected_username.encode("utf-8")
        )
        password_ok = hmac.compare_digest(
            password.encode("utf-8"), expected_password.encode("utf-8")
        )
        return user_ok and password_ok

    def _request_auth(self) -> None:
        """Send a 401 Basic challenge and close the connection."""

        logger.debug(
            "preview auth rejected request from %s",
            self.client_address[0],
        )
        self.send_response(401)
        self.send_header(
            "WWW-Authenticate",
            'Basic realm="Chandelier", charset="UTF-8"',
        )
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        # The request body (if any) was never read; closing avoids
        # leaving unread bytes on a reused keep-alive connection.
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_GET(self) -> None:
        """Dispatch read-only routes after the auth check."""

        # Authenticate before touching the path, gallery or filesystem.
        if not self._authorized():
            self._request_auth()
            return
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
        elif path == "/gallery":
            self._gallery()
        elif path == "/api/gallery":
            self._gallery_items()
        elif path.startswith("/thumb/"):
            self._thumbnail(path[len("/thumb/"):])
        elif path.startswith("/view/"):
            self._view(path[len("/view/"):])
        elif path.startswith("/media/"):
            self._media(path[len("/media/"):])
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        """Dispatch remote-control and gallery-delete routes."""

        if not self._authorized():
            self._request_auth()
            return
        path = urlsplit(self.path).path
        if path == "/control":
            self._control()
        elif path == "/screen":
            self._screen()
        elif path == "/touch":
            self._touch()
        elif path == "/api/gallery/delete":
            self._delete_gallery_items()
        else:
            self.send_error(404, "Not Found")

    def _send_body(self, body: bytes, content_type: str) -> None:
        """Send a complete in-memory body with an explicit length."""

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        # An explicit length lets the browser show progress and avoids
        # waiting for EOF to delimit the body.
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # One-shot response: close so the per-request thread is released
        # without keep-alive bookkeeping.
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def _snapshot(self) -> None:
        """Serve the newest composed frame as a JPEG, if one exists."""

        jpeg = self.server.preview.snapshot()
        if jpeg is None:
            # 503, not 404: the route exists but the camera has not
            # produced a frame yet.
            self.send_error(503, "No frame available")
            return
        self._send_body(jpeg, "image/jpeg")

    def _gallery(self) -> None:
        """Serve the gallery page markup."""

        self._send_body(_gallery_html(), "text/html; charset=utf-8")

    def _gallery_items(self) -> None:
        """Return the listing as JSON for the gallery page."""

        query = parse_qs(urlsplit(self.path).query)
        # ``?refresh=1`` bypasses the listing TTL after a manual reload.
        items = self.server.preview.gallery.items(force="refresh" in query)
        self._json_response(
            200,
            {
                "count": len(items),
                "items": [item.as_payload() for item in items],
            },
        )

    def _delete_gallery_items(self) -> None:
        """Delete a bounded batch of gallery IDs."""

        # IDs are resolved against the scanned listing in WebGallery; the
        # request itself never carries filesystem paths.  Bound both the body
        # size and the batch count so one request cannot stall the server.
        data = self._read_json(limit=65536)
        ids = data.get("ids") if isinstance(data, dict) else None
        if not isinstance(ids, list):
            self._json_response(400, {"ok": False})
            return
        ids = [value for value in ids if isinstance(value, str)][:200]
        if not ids:
            self._json_response(400, {"ok": False})
            return

        deleted, failed = self.server.preview.gallery.delete_items(ids)
        self._json_response(
            200,
            {"ok": True, "deleted": deleted, "failed": failed},
        )

    @staticmethod
    def _query_size(query: dict, key: str) -> int:
        """Read a thumbnail dimension, defaulting to 400px."""

        try:
            return int(query.get(key, ["400"])[0])
        except (TypeError, ValueError):
            # A malformed w/h must not blank out the whole grid.
            return 400

    def _thumbnail(self, raw_name: str) -> None:
        """Serve a cached, cover-cropped JPEG thumbnail."""

        query = parse_qs(urlsplit(self.path).query)
        jpeg = self.server.preview.gallery.thumbnail(
            unquote(raw_name),
            self._query_size(query, "w"),
            self._query_size(query, "h"),
        )
        if jpeg is None:
            self.send_error(404, "Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        # Thumbnail URLs carry a ?v= version token, so a long browser
        # cache is safe; "private" keeps them out of shared proxies.
        self.send_header("Cache-Control", "private, max-age=600")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(jpeg)

    def _view(self, raw_name: str) -> None:
        """Serve an image the browser can display in the lightbox."""

        item_id = unquote(raw_name)
        item = self.server.preview.gallery.find(item_id)
        if item is None or item.media_type != "image":
            self.send_error(404, "Not Found")
            return
        if item.extension in BROWSER_IMAGE_EXTENSIONS:
            # The original is already browser-decodable; stream it as-is.
            self._media(raw_name)
            return

        # TIFF and similar formats need a JPEG rendition for an <img> tag.
        jpeg = self.server.preview.gallery.render_image(item_id)
        if jpeg is None:
            self.send_error(404, "Not Found")
            return
        self._send_body(jpeg, "image/jpeg")

    def _media(self, raw_name: str) -> None:
        """Stream a listed file, honouring one HTTP byte range."""

        item, handle = self.server.preview.gallery.open_media(
            unquote(raw_name)
        )
        if item is None or handle is None:
            self.send_error(404, "Not Found")
            return

        with handle:
            # Take the size from the open descriptor: the path could be
            # replaced between the listing lookup and the open.
            file_size = os.fstat(handle.fileno()).st_size
            start, end = 0, file_size - 1
            partial = False
            range_header = self.headers.get("Range")
            # Ignore unsupported range units; only bytes are meaningful here.
            if range_header and range_header.strip().lower().startswith(
                "bytes="
            ):
                parsed = _parse_byte_range(range_header, file_size)
                if parsed is None:
                    # RFC 7233: an unsatisfiable range gets 416 plus
                    # ``Content-Range: bytes */size`` so the client can
                    # recover by requesting the whole file.
                    self.send_response(416)
                    self.send_header(
                        "Content-Range", f"bytes */{file_size}"
                    )
                    self.send_header("Content-Length", "0")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    return
                start, end = parsed
                partial = True

            length = max(0, end - start + 1)
            content_type = (
                mimetypes.guess_type(item.name)[0]
                or "application/octet-stream"
            )
            # 206 for a partial body, 200 for the whole file.
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            # Advertise range support so browsers can seek video.
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header(
                    "Content-Range", f"bytes {start}-{end}/{file_size}"
                )
            # RFC 5987 filename* is the portable way to send non-ASCII
            # names without mangling them.
            self.send_header(
                "Content-Disposition",
                "inline; filename*=UTF-8''" + quote(item.name, safe=""),
            )
            # Media may be auth-protected and can change; keep it private
            # and short-lived rather than letting proxies hold it.
            self.send_header("Cache-Control", "private, max-age=300")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            # Seek so a Range request reads only the bytes it asked for.
            handle.seek(start)
            remaining = length
            while remaining > 0:
                # Bounded chunks keep a large video out of RAM.
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # The player seeked or closed the tab; stop quietly.
                    break
                remaining -= len(chunk)

    def _client_gone(self) -> bool:
        """Detect a closed peer without writing (used while idle)."""

        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
        except (OSError, ValueError):
            return True
        if not readable:
            return False
        try:
            # MSG_PEEK leaves the byte queued; a zero-length peek is EOF.
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _stream(self) -> None:
        """Stream the latest frame as multipart/x-mixed-replace MJPEG."""

        preview = self.server.preview
        stopping = preview.stop_event
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store, no-cache")
        # Pragma helps HTTP/1.0 proxies that ignore Cache-Control.
        self.send_header("Pragma", "no-cache")
        # The stream has no declared length, so closing is the only way
        # to end it; this also releases the handler thread.
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        preview.client_open()
        # -1 forces the first available frame to be sent immediately,
        # even if it was published before the client connected.
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
                # wait_for_frame can return the frame already sent when
                # the wait times out; skip it and poll the peer instead.
                if current == sequence:
                    if self._client_gone():
                        break
                    continue
                jpeg = preview.jpeg_for(current, frame)
                if jpeg is None:
                    continue
                sequence = current

                # The first part needs a leading boundary; later parts
                # reuse the one written after the previous frame.
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
            # The peer vanished mid-write; the finally block still keeps
            # the client count accurate.
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
        # SSE is an unbounded stream; closing ends it and frees the
        # handler thread.
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
                # Skip the frame already delivered after a wait timeout.
                if current == sequence:
                    if self._client_gone():
                        break
                    continue
                jpeg = preview.jpeg_for(current, frame)
                if jpeg is None:
                    continue
                sequence = current
                # SSE fields are text; base64 keeps the JPEG bytes
                # intact through UTF-8 decoding in the browser.
                self.wfile.write(b"data: ")
                self.wfile.write(base64.b64encode(jpeg))
                # A blank line terminates the event.
                self.wfile.write(b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass
        finally:
            preview.client_close()

    def _status(self) -> None:
        """Return server state and the active theme for page pollers."""

        preview = self.server.preview
        self._json_response(
            200,
            {
                "enabled": preview.enabled,
                "port": preview.port,
                "clients": preview.client_count,
                "theme": _theme_payload(),
            },
        )

    def _control(self) -> None:
        """Handle a remote button press from the preview page."""

        data = self._read_json()
        action = data.get("action") if isinstance(data, dict) else None
        ok = bool(action) and self.server.preview.send_action(str(action))
        self._json_response(200 if ok else 400, {"ok": ok})

    def _screen(self) -> None:
        """Handle a screen tab selection."""

        data = self._read_json()
        screen = data.get("id") if isinstance(data, dict) else None
        ok = bool(screen) and self.server.preview.switch_to(str(screen))
        self._json_response(200 if ok else 400, {"ok": ok})

    def _touch(self) -> None:
        """Handle a tap on the preview image."""

        data = self._read_json()
        x = data.get("x") if isinstance(data, dict) else None
        y = data.get("y") if isinstance(data, dict) else None
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            self._json_response(400, {"ok": False})
            return
        # Coordinates are normalised so the page does not need to know
        # the device's 320x240 layout; the controller scales them.
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            self._json_response(400, {"ok": False})
            return
        ok = self.server.preview.send_touch(x, y)
        self._json_response(200 if ok else 400, {"ok": ok})

    def _read_json(self, limit: int = 4096):
        """Read a bounded JSON body; ``None`` for missing or bad input."""

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            return None
        # Without a valid length there is no safe place to stop reading;
        # reject empty and oversized bodies before allocating.
        if not 0 < length <= limit:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except (TypeError, ValueError):
            return None

    def _json_response(self, status: int, payload: dict) -> None:
        """Send a small JSON object and close the connection."""

        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Polling endpoints must never be answered from a stale cache.
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)
