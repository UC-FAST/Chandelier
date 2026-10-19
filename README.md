# Chandelier

Raspberry Pi camera terminal with a 320×240 display, physical controls,
gallery, system monitor and multi-source sound localisation. A small web
preview server streams the device screen and provides remote controls.

## Features

- Camera capture and video recording with an on-device menu
- Shutter presets from 1/8000 to 30s; presets longer than 1/15s are used for
  stills only so the live preview stays responsive
- HDR modes require Auto Exposure; turning Auto Exposure off clears HDR
- Gallery browsing from local storage
- System monitor pages: hardware, power, network and services
- Direction-of-arrival estimation (SRP-PHAT + DUET) with Silero VAD
- Web preview with MJPEG/SSE streaming, an on-screen remote, menu touch
  selection and live theme synchronisation
- Menu themes: Dark Pro, High Contrast, Industrial, Classic, Ubuntu,
  Liquid Glass, Windows 2000, Windows 7 and Aero
- Battery, power and RTC sensors
- Runs as a systemd service (`cam.service`)

## Hardware

Tested on a Raspberry Pi 5 running Debian 13 (trixie):

- 320×240 ST7735 TFT display
- libcamera / Picamera2 camera
- Seeed ReSpeaker 4 Mic Array (UAC1.0, USB `2886:0018`); the localizer uses
  the four raw microphone channels of its 6-channel, 16 kHz stream
- GPIO buttons and rotary encoder
- BQ32002 RTC, MAX17048 battery gauge and INA230 power monitor

## Install

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-pi.txt
```

`requirements.txt` contains the platform-independent dependencies;
`requirements-pi.txt` adds the Raspberry Pi hardware packages. The
localisation pipeline additionally needs `torch`, `speechbrain`, `PyAudio`,
`soundfile` and `plotly` (see `requirements.txt`).

The project can also be installed as a package:

```bash
pip install -e .
```

## Run

Foreground:

```bash
python3 startup.py
```

As a service:

```bash
sudo systemctl status cam.service
sudo systemctl restart cam.service
```

`cam.service` runs `startup.py` as the `pi` user with
`WorkingDirectory=/home/pi/chandelier-deepseek`. After `pip install -e .` the
`chandelier` console script is available as well.

## Configuration

- `config/config.json` — GPIO pins, camera, screen, LEDs, sensors, localizer
  and marker settings, NTP server and logging
- `config/menus.json` — menu tree and persisted user selections (including
  the active theme). Old menus are migrated on startup
- `config/pisp/*.json` — libcamera tuning files
- `chandelier/themes.py` — theme registry; the menu theme list is derived
  from this registry

## Web preview

Enable the server in the device menu under **System → Web Server**. The
default port is `5989`; open `http://<pi-ip>:5989/` from a phone or computer
on the same network.

Endpoints:

| Path | Purpose |
|------|---------|
| `/` | Remote-control page |
| `/events` | SSE stream of base64 JPEG frames |
| `/snapshot.jpg` | Latest frame as JPEG |
| `/status` | Server state and active theme |
| `/control` | Send a button action |
| `/screen` | Switch the displayed screen |
| `/touch` | Tap a menu row (normalised `x`, `y`) |
| `/gallery` | Web gallery page (thumbnail grid and viewer) |
| `/api/gallery` | Media listing as JSON; `?refresh=1` rescans |
| `/api/gallery/delete` | Delete items (`POST {"ids": [...]}`) |
| `/thumb/<id>` | Cover-cropped JPEG thumbnail (`w`, `h` query) |
| `/view/<id>` | Browser-friendly image for the lightbox |
| `/media/<id>` | Original photo or video (supports HTTP Range) |

The page follows the active device theme. Liquid Glass uses translucent
panels with `backdrop-filter` over a blurred live frame.

The gallery button in the header (and the **Gallery preview** chip below the
live frame) opens a thumbnail grid of the media directories. Item IDs are
bare filenames for the photo directory (`camera.path`, default `pict`) and
are prefixed with the directory name for recordings (`camera.video_path`,
default `pict/video`, so `video/<name>`). Thumbnails are generated on demand;
clicking a tile opens a viewer with keyboard and swipe navigation, plus a
download link. Images the browser cannot decode (TIFF) are re-encoded for
the viewer; videos are served with HTTP Range support so the browser can
seek. The grid rescans on load and on the refresh button, and polls every
15 seconds, so new photos and recordings appear without restarting the
service. The **Manage** button enables multi-select deletion; the viewer
also has a Delete action. Deletions remove the file and its device-side
rendered cache.

### Authentication

By default the server is unauthenticated and intended for a trusted LAN. To
require HTTP Basic Auth, create `config/preview_auth.json`:

```json
{
  "username": "pi",
  "password": "change-me"
}
```

Protect the file and restart the service:

```bash
chmod 600 config/preview_auth.json
sudo systemctl restart cam.service
```

The file is excluded from git. Delete it to disable authentication again.
Basic Auth sends credentials on every request, so use HTTPS or a VPN
(Tailscale, WireGuard) when accessing the server from outside the LAN. The
server prompts for credentials and applies them to all endpoints, including
the SSE stream and the control/touch endpoints.

## Tests

```bash
python3 -m pytest -q
```

The suite covers the menu system, themes, rendering, preview server,
localizer geometry and hardware adapters. Tests that need the separate
standalone game package are skipped when it is not installed.

## Project layout

```
chandelier/            application package
  app.py               entry point and end wiring
  config.py            configuration loading
  controller.py        input dispatch and screen switching
  menu_defaults.py     menu schema migrations
  themes.py            theme registry
  hardware/            display, camera, GPIO, LEDs, microphone, sensors
  localizer/           SRP-PHAT/DUET localisation and VAD
  media/               gallery and media handling
  services/            preview server, system stats, network, time sync
  ui/                  camera, gallery, menu and system-monitor screens
  cli/                 offline/online localisation tools
config/                runtime configuration and camera tuning
tests/                 pytest suite
```

## License

MIT — see [LICENSE](LICENSE).
