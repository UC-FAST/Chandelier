#!/usr/bin/env python3
from multiprocessing.managers import ListProxy

from controlledEnd import MenuControlledEnd, GalleryControlledEnd, CameraControlledEnd, SystemMonitor
from components import lcd20
import universalControl
from utils import SystemTimeManager
from utils import ConfigLoader
import logging
import os
import sys
import multiprocessing

from multi_ssl.mic import MicrophoneStream
from srp_phat_online import main
from utils.AngleToPixelMapper import aziel_to_screen_spherical


sys.path.append('./')
sys.path.append('./components/')
sys.path.append('./utils/')


ConfigLoader(os.path.abspath('./config.json'))

timestamp = SystemTimeManager().get_timestamp()
SystemTimeManager().set_system_time_with_timestamp(timestamp)

tuning = './pisp/imx477.json'
config = ConfigLoader('./config.json')


def getC(q: ListProxy):
    microphone_stream = MicrophoneStream(
        rate=16000,
        chunk=1600,
        channels=6,
        ignored_channels=[0, 5],
        device=0
    )
    for i in main(3, microphone_stream):
        if i[0] > 0.03:
            azi, ele = tuple(round(x) for x in i[1].squeeze().tolist())
            q[:] = [i[0], aziel_to_screen_spherical(azi, ele)]
            print(f'azi={azi},ele={ele} =>',q[1])
        else:
            print(i[0])


if __name__ == "__main__":
    print(sys.argv)
    q = multiprocessing.Manager().list([None, None])
    p1 = multiprocessing.Process(target=getC, args=(q,))
    p1.start()

    u = universalControl.UniversalControl(
        lcd20.Lcd(),
        [
            CameraControlledEnd(
                verbose_console=logging.FATAL,
                tuning_file_path=tuning
            ),
            SystemMonitor(),
            MenuControlledEnd(
                path='a.json',
                show_preview=True,
                row_count=5,
                show_index=True,
                font_height=14,
                padding=(5, 5, 5, 5)
            ),
            GalleryControlledEnd(pictPath=config['camera']['path']),
        ],
        argv=sys.argv
    )

    try:
        u.main_loop(q)
    finally:
        if p1.is_alive():
            p1.terminate()
            p1.join()
