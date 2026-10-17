import types

import numpy as np

from chandelier.localizer.vad import get_microphone_chunks


class FakeStream:
    rate = 16000

    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        yield from self.chunks


class FakeModel:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self, chunk, sample_rate):
        value = next(self.values)
        return types.SimpleNamespace(item=lambda: value)



def test_chunks_accumulate_and_yield():
    chunks = [np.ones((512, 4), dtype=np.float32) for _ in range(4)]
    stream = FakeStream(chunks)
    model = FakeModel([0.9, 0.9, 0.9, 0.9])

    output = list(
        get_microphone_chunks(
            stream,
            min_to_cumulate=2,
            max_to_cumulate=3,
            deque_size=2,
            speech_threshold=0.5,
            model=model,
        )
    )

    assert len(output) == 1
    assert output[0].sample_rate == 16000
    assert output[0].waveform.shape == (512 * 3, 4)
    assert stream.closed is True


def test_settings_provider_drives_threshold_and_chunk_limit():
    chunks = [np.ones((512, 4), dtype=np.float32) for _ in range(3)]
    stream = FakeStream(chunks)
    model = FakeModel([0.9, 0.9, 0.9])

    class Settings:
        speech_threshold = 0.5
        max_to_cumulate = 2

    output = list(
        get_microphone_chunks(
            stream,
            max_to_cumulate=10,
            speech_threshold=0.95,
            model=model,
            settings_provider=lambda: Settings(),
        )
    )

    assert len(output) == 1
    assert output[0].waveform.shape == (512 * 2, 4)


def test_stop_event_ends_iteration():
    chunks = [np.ones((512, 4), dtype=np.float32) for _ in range(3)]

    class Stop:
        def is_set(self):
            return True

    output = list(
        get_microphone_chunks(
            FakeStream(chunks),
            model=FakeModel([0.9, 0.9, 0.9]),
            stop_event=Stop(),
        )
    )
    assert output == []
