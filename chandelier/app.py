"""Application composition root."""

from __future__ import annotations

import logging
import sys

from chandelier.config import AppConfig, load_config
from chandelier.controller import Controller
from chandelier.log import configure_logging
from chandelier.media.library import MediaBrowser
from chandelier.ui import (
    CameraControlledEnd,
    GalleryControlledEnd,
    MenuControlledEnd,
    SystemMonitor,
)

logger = logging.getLogger(__name__)


def build_ends(
    config: AppConfig,
    camera,
    leds,
    library,
    stats,
    localizer=None,
    preview_server=None,
):
    """Build the four screens in their original order."""

    return [
        CameraControlledEnd(
            camera,
            leds,
            config,
            localizer=localizer,
            preview_server=preview_server,
            stats=stats,
        ),
        SystemMonitor(
            stats,
            config,
            preview_server=preview_server,
            localizer=localizer,
            camera=camera,
        ),
        MenuControlledEnd(
            path=config.menus_file,
            width=config.screen.width,
            height=config.screen.height,
            show_preview=True,
            row_count=5,
            show_index=True,
            font_height=14,
            padding=(5, 5, 5, 5),
        ),
        GalleryControlledEnd(library, config),
    ]


def main(argv: list[str] | None = None) -> int:
    """Compose the application, run the controller and clean up.

    Returns a process exit code: 0 after a normal stop, 1 when startup or
    the render loop fails.
    """

    argv = list(sys.argv if argv is None else argv)

    config = load_config()
    configure_logging(config.debug_level)
    logger.info("starting chandelier")

    localizer = None
    closeables = []

    def close_all() -> None:
        # Reverse construction order so consumers stop before the resources
        # they depend on (e.g. the camera before the display it draws to).
        for close in reversed(closeables):
            try:
                close()
            except Exception:
                logger.debug("cleanup failed", exc_info=True)

    try:
        # Start the localiser before any real hardware so the child process
        # does not inherit GPIO/SPI/camera file descriptors.
        try:
            from chandelier.localizer.worker import start_worker

            localizer = start_worker(config.localizer, config.frame_size)
            closeables.append(localizer.stop)
        except Exception:
            logger.exception(
                "sound localisation unavailable; continuing without it"
            )

        from chandelier.hardware.camera import CameraDevice
        from chandelier.hardware.display import St7735Display
        from chandelier.hardware.gpio_input import GpioInputPanel
        from chandelier.hardware.leds import Leds
        from chandelier.hardware.sensors.bq32002 import BQ32002
        from chandelier.hardware.sensors.ina230 import INA230
        from chandelier.hardware.sensors.max17048 import MAX17048
        from chandelier.services.network import Network
        from chandelier.services.preview_server import (
            PreviewServer,
            load_auth_file,
        )
        from chandelier.services.system_stats import SystemStats
        from chandelier.services.time_sync import SystemTimeManager

        display = St7735Display(
            config.screen.width, config.screen.height, config.pin
        )
        closeables.append(display.close)
        display.init()

        leds = Leds(config.led)
        closeables.append(leds.close)

        camera = CameraDevice(
            config.camera,
            config.screen,
            verbose_console=logging.FATAL,
        )
        closeables.append(camera.close)

        rtc = BQ32002(config.sensors.bq32002.bus)
        closeables.append(rtc.close)
        battery = MAX17048(config.sensors.max17048.bus)
        closeables.append(battery.close)
        power_monitor = INA230(config.sensors.ina230)
        closeables.append(power_monitor.close)

        library = MediaBrowser(
            str(config.camera.path),
            config.screen.width,
            config.screen.height,
        )
        network = Network(config.internet_test_url)
        stats = SystemStats(battery, power_monitor, rtc, network)

        preview_server = PreviewServer(
            auth=load_auth_file(config.preview.auth_file),
            media_path=[config.camera.path, config.camera.video_path],
        )
        closeables.append(preview_server.stop)

        ends = build_ends(
            config,
            camera,
            leds,
            library,
            stats,
            localizer=localizer,
            preview_server=preview_server,
        )
        controller = Controller(
            display, ends, config, argv, preview_server=preview_server
        )
        # The web preview drives the same controller entry points as the
        # physical buttons, so no separate command path is needed.
        preview_server.set_action_handler(controller.handle_action)
        preview_server.set_screen_handler(controller.switch_to)
        preview_server.set_touch_handler(controller.handle_touch)
        input_panel = GpioInputPanel(config.pin, controller.handle_action)
        closeables.append(input_panel.close)

        time_manager = SystemTimeManager(rtc, network, config.ntp_server)
        # Time sync is best effort: the UI must start even without a network
        # or when changing the clock needs privileges the service lacks.
        try:
            timestamp = time_manager.get_timestamp()
            time_manager.set_system_time_with_timestamp(timestamp)
        except Exception:
            logger.exception("time synchronisation failed")

        controller.main_loop(
            localizer.latest if localizer is not None else None
        )
        return 0
    except Exception:
        logger.exception("application terminated with an error")
        return 1
    finally:
        close_all()
        logger.info("chandelier stopped")
