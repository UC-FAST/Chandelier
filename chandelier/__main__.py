"""Allow ``python -m chandelier`` to start the application."""

from chandelier.app import main

if __name__ == "__main__":
    raise SystemExit(main())
