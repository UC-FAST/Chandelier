#!/usr/bin/env python3
"""Backwards-compatible entry point used by ``cam.service``.

Keep this file at the project root: the systemd unit executes
``python3 startup.py`` with the project directory as its working directory.
All real work happens in :func:`chandelier.app.main`.
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    # Allow ``python3 /path/to/startup.py`` from any working directory.
    project_root = str(Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from chandelier.app import main

    raise SystemExit(main(sys.argv))
