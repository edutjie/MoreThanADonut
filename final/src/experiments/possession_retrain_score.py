'Choose inference settings from external out-of-fold predictions only.'


import numpy as np


def native_probabilities(p, length, fps):
    samples = np.arange(len(p)) * fps / 5
    native = np.column_stack([np.interp(np.arange(length), samples, side) for side in p.T])
    return np.round(native, 4)
