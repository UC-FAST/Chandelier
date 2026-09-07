import torch
import numpy as np
from multi_ssl.mic.microphone_stream import MicrophoneStream
from multi_ssl.utils import Duet, doa_detection
from multi_ssl.mic import get_microphone_chunks, init_respeaker_mic_array


def main(src,microphone_stream):
    for sample_rate, waveform in get_microphone_chunks(
        microphone_stream,
        min_to_cumulate=2,
        max_to_cumulate=2,
        speech_threshold=0.6,
    ):
        
        duet = Duet(
            waveform.transpose(),
            n_sources=src,
            sample_rate=sample_rate,
            delay_max=0.0,
            n_delay_bins=50,
            output_all_channels=True,
        )

        estimates = duet().astype(np.float32)
        doas = doa_detection(
            init_respeaker_mic_array(),
            torch.from_numpy(estimates),
            sample_rate=sample_rate,
        )
        doas[doas[:, 0] < 0] += torch.FloatTensor([[360, 0]])
        yield np.max(waveform), doas
            



if __name__ == "__main__":
    microphone_stream = MicrophoneStream(
        rate=16000,
        chunk=1600,
        channels=6,
        ignored_channels=[0, 5],
        device=0
    )
    for i in main(3,microphone_stream):
        if i[0]>0.01:
            print(i)
