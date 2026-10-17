import numpy as np
from icecream import ic
from scipy.io.wavfile import read, write
from multi_ssl.utils.bss import Duet

if __name__ == "__main__":

    fs, x = read("/home/pi/et2026/data/a180e20_a225e35_a270e50/250cm/a180e19_a224e34_a269e49_3_1c186780.wav")

    x = np.transpose(x)[1:5, :]
    ic(x.shape)
    duet = Duet(
        x,
        n_sources=3,
        sample_rate=fs,
        delay_max=2.0,
    )
    estimates = duet()
    ic(estimates.shape)
    for i in range(duet.n_sources):
        write(f"duet_s{i}.wav", duet.fs, estimates[i, :] + 0.05 * duet.x1)
    duet.plot_atn_delay_hist()
