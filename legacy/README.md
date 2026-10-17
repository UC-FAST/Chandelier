# Legacy archive

This directory contains code that is no longer imported by the application.
It is kept for reference and historical comparison only. Nothing here is
packaged, tested, or maintained.

## Layout

- `old/` — the complete pre-0.2 source tree:
  - `old/utils/`, `old/components/`, `old/controlled_end/`,
    `old/frame_decorator/`, `old/multi_ssl/`
  - `old/universal_control.py`
  - `old/requirements-py311.txt`, `old/requirements-py313.txt`
- `utils/` — standalone/experimental utilities removed from the runtime:
  - `deepseek.py` (Ollama demo, not part of the device)
  - `tuning.py` (ReSpeaker XMOS USB tuning tool, needs `pyusb`)
  - `logger_config.py` (unused and contained an invalid zero-width character)
  - `effect.py` (unused HDR helper)
  - `bkm.py` (ReSpeaker parameter snapshot)
  - `system_mon.py` (manual sensor polling script)
- `frame_decorator/` — unused widgets: `main.py`, `dialog_box.py`,
  `direction_indicator.py`, `hist.py`
- `tests_manual/` — old manual experiments (`test_duet.py`,
  `test_microphone.py`, `srp_phat_offline.py`, `srp_phat_online.py`,
  `srp_visualizer.py`). They may reference `/home/pi/...` paths and old
  dependency versions. Their replacements live in `chandelier/cli/`.
- `colors_png/c/` — unused PNG colour swatches.
- `a.json` — the pre-0.2 flat menu configuration. The runtime now uses
  `config/menus.json` with nested end groups and snake_case keys.
- `camlimit` — old camera limit file.
- `old/components/readme.md` — old component documentation.

The old code is intentionally not fixed. In particular, `logger_config.py`
contains a non-ASCII zero-width character and several scripts depend on
packages that are no longer installed.
