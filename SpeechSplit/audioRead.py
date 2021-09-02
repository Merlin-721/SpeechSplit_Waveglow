import numpy as np
from .hparams import hparams

MAX_WAV_VALUE = 32768


def get_id(speaker,speaker_ids):
    spkr_list = list(speaker_ids.keys())
    spkr_list = sorted(spkr_list, key=lambda item: (int(item.partition(' ')[0])
                            if item[0].isdigit() else float('inf'), item))

    spkid = np.zeros((hparams.dim_spk_emb,), dtype=np.float32)
    spkid[spkr_list.index(speaker)] = 1.0

    return spkid
