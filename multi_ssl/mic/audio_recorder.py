"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.

Modified to the latest version since some of the functions were deprecated.

Original code repository: 
    https://github.com/pytorch/audio/blob/master/examples/interactive_asr/vad.py#L38
"""

import numpy as np
import torch
from typing import Generator, Tuple
from collections import deque

from multi_ssl.mic.microphone_stream import MicrophoneStream

torch.set_num_threads(1)


def get_microphone_chunks(
    microphone_stream: MicrophoneStream,
    *,
    min_to_cumulate=5,
    max_to_cumulate=20,
    deque_size=5,
    speech_threshold=0.8,
    sample_rate=16000
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Real-time VAD(voice-activity-detection) chunk catcher.

    Augments
    --------
    microphone_stream : MicrophoneStream
        the microphone stream object.
    min_to_cumulate : int
        the minimum length of waveform, default is 5 chunks.
    max_to_cumulate : int
        the maximun length of waveform, default is 20 chunks.
    precumulate : int
        the length of waveform to keep before the speech, default is 5 chunks.
    speech_threshold : float
        the threshold to decide if the chunk contains speech, default is 0.8.

    Returns
    -------
    (waveform, sample_rate) : Tuple[ndarray, int]

    Example
    -------
    >>> import soundfile as sf
    >>> import numpy as np
    >>> from multi_ssl.mic.microphone_stream import MicrophoneStream
    >>> from multi_ssl.mic.audio_recorder import get_microphone_chunks
    >>> waveforms = []
    >>> microphone_stream = MicrophoneStream(
    >>>     rate=16000,
    >>>     chunk=1600,
    >>>     channels=6,
    >>>     ignored_channels=[0, 5],
    >>> )
    >>> try:
    >>>     for sample_rate, waveform in get_microphone_chunks(
    >>>         microphone_stream,
    >>>         precumulate=0,
    >>>     ):
    >>>         print(sample_rate, waveform.shape)  # 16000, (1600, 4)
    >>>         waveforms.append(waveform)
    >>> finally:
    >>>     sf.write("output.wav", np.concatenate(waveforms, axis=0), 16000, "PCM_16")
    """

    cumulated = list()
    precumulated = deque(maxlen=deque_size)

    model, _ = torch.hub.load(
        repo_or_dir="/home/pi/.cache/torch/hub/snakers4_silero-vad_master",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
        source="local"
    )

    # NOTE: silero-vad configs
    chunk_size = 512 if microphone_stream.rate == 16000 else 256
    with microphone_stream as stream:
        for chunk in stream:
            confidence = model(
                torch.from_numpy(np.copy(chunk[:chunk_size, 0])),
                stream.rate,
            ).item()
            is_speech = confidence >= speech_threshold

            # ---------- 分状态缓存音频 ----------
            # 情况1：当前是语音 OR 已经处于语音累积中 → 加入累积列表
            # （即使这一帧刚好没检测到语音，但已经在说话段里也继续累积，防止中间短暂停顿被切断）

            if is_speech or cumulated:
                cumulated.append(chunk)
            else:
                precumulated.append(chunk)

            # ---------- 判断是否输出一段完整语音 ----------
            # 触发输出的两个条件（满足其一即可）：
            # 条件A：当前帧不是语音 + 已经累积了足够多chunk → 语音结束，输出整段
            # 条件B：累积chunk数达到上限 → 强制截断输出，防止内存无限增长
            if (not is_speech and len(cumulated) >= min_to_cumulate) or len(cumulated) >= max_to_cumulate:
                waveform = np.concatenate(
                    (list(precumulated) + cumulated), axis=0) # 把【预缓存的前置音频】 + 【累积的语音主体】沿时间轴拼接成完整波形
                yield (stream.rate, waveform)
                cumulated.clear()
                precumulated.clear()
