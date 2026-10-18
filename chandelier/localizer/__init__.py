"""Sound-source localisation (DUET separation + SRP-PHAT DOA estimation).

Heavy dependencies (torch, scipy, plotly) are imported lazily inside the
functions that need them so that importing this package stays cheap and
works on machines without the localisation extras installed.
"""

from chandelier.localizer.bss import Duet
from chandelier.localizer.geometry import mic_array_positions
from chandelier.localizer.peaks import find_peak_indices
from chandelier.localizer.ssl import doa_detection

__all__ = [
    "Duet",
    "doa_detection",
    "find_peak_indices",
    "mic_array_positions",
]
