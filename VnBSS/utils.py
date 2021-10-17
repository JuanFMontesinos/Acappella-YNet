import numpy as np


def normalize_max(waveform):
    if np.abs(waveform).max() != 0:
        waveform_out = waveform / np.abs(waveform).max()
    else:
        waveform_out = waveform

    return waveform_out
