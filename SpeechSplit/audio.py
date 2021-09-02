import librosa.filters
import numpy as np
from hparams import hparams
from nnmnkwii import preprocessing as P


def trim(quantized):
    start, end = start_and_end_indices(quantized, hparams.silence_threshold)
    return quantized[start:end]


def preemphasis(x, coef=0.85):
    return P.preemphasis(x, coef)


def inv_preemphasis(x, coef=0.85):
    return P.inv_preemphasis(x, coef)


def adjust_time_resolution(quantized, mel):
    """Adjust time resolution by repeating features

    Args:
        quantized (ndarray): (T,)
        mel (ndarray): (N, D)

    Returns:
        tuple: Tuple of (T,) and (T, D)
    """
    assert len(quantized.shape) == 1
    assert len(mel.shape) == 2

    upsample_factor = quantized.size // mel.shape[0]
    mel = np.repeat(mel, upsample_factor, axis=0)
    n_pad = quantized.size - mel.shape[0]
    if n_pad != 0:
        assert n_pad > 0
        mel = np.pad(mel, [(0, n_pad), (0, 0)], mode="constant", constant_values=0)

    # trim
    start, end = start_and_end_indices(quantized, hparams.silence_threshold)

    return quantized[start:end], mel[start:end, :]


def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def get_hop_size():
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size

