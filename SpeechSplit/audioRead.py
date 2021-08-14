import os
import sys
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk #signal processing toolkit
import pickle
try:
    from .utils import butter_highpass, speaker_normalization, pySTFT
    from .hparams import hparams
except:
    from utils import butter_highpass, speaker_normalization, pySTFT
    from hparams import hparams

SAMPLE_RATE = hparams.sample_rate
MAX_WAV_VALUE = 32768

def read_audio(dirName,speaker,recording):
    # read audio file
    b, a = butter_highpass(30, SAMPLE_RATE, order=5)

    x, fs = sf.read(os.path.join(dirName,speaker,recording))
    assert fs == SAMPLE_RATE
    if x.shape[0] % hparams.hop_size == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = signal.filtfilt(b, a, x)
    prng = RandomState(int(speaker[1:])) 
    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
    return wav

def gen_spect(wav):
    mel_basis = mel(SAMPLE_RATE, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))

    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = (D_db + 100) / 100        
    return S

def gen_f0(wav, lo, hi):
    f0_rapt = sptk.rapt(wav.astype(np.float32)*MAX_WAV_VALUE, SAMPLE_RATE, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
    return f0_norm

def get_id(speaker,speaker_ids):
    spkr_list = list(speaker_ids.keys())
    spkr_list = sorted(spkr_list, key=lambda item: (int(item.partition(' ')[0])
                            if item[0].isdigit() else float('inf'), item))

    spkid = np.zeros((hparams.dim_spk_emb,), dtype=np.float32)
    spkid[spkr_list.index(speaker)] = 1.0

    return spkid

def gen_data(dirName, subDirList,spk2gen):
    # For each speaker eg p226
    mels, f0s, pIDs, embeddings = [], [], [], []

    for speaker in sorted(subDirList):
        print(speaker)
        
        if spk2gen[speaker] == 'M':
            lo, hi = 50, 250
        elif spk2gen[speaker] == 'F':
            lo, hi = 100, 600
        else:
            raise ValueError("Speaker not in list")


        _,_, fileList = next(os.walk(os.path.join(dirName,speaker)))
        # For each recording from speaker
        for recording in sorted(fileList):
            spk_embed = get_id(speaker, spk2gen)

            wav = read_audio(dirName,speaker,recording) 
            # compute spectrogram
            S = gen_spect(wav) 
            # extract f0
            f0_norm = gen_f0(wav,lo,hi)
            # assert len(S) == len(f0_rapt)
            assert len(S) == len(f0_norm)
                
            mels.append(S.astype(np.float32))
            f0s.append(f0_norm.astype(np.float32))
            pIDs.append(speaker)
            embeddings.append(spk_embed)

    return mels, f0s, pIDs, embeddings

def make_rows(pIDs, embeddings, mels, f0s):
    M = len(pIDs)
    uIDs = range(M)
    rows = []
    for i in range(M):
        row = [pIDs[i], embeddings[i], (mels[i], f0s[i], len(mels[i]), str(uIDs[i]))]
        rows.append(row)
    return rows


if __name__ == '__main__':
    rootDir = "assets/wavs/"
    outputDir = "processed_assets/"


    # speaker to gender - dictionary of pXXX: M/F
    spk2gen = pickle.load(open('assets/spk2gen.pkl', "rb"))

    dirName, subDirList, _ = next(os.walk(rootDir))

    mels, f0s, pIDs, embeddings = gen_data(dirName, subDirList, spk2gen)

    # [pID, [speaker embed], [mel, f0, len, uid]]
    rows = make_rows(pIDs, embeddings, mels, f0s)

    outfile = open(str(outputDir + "customData.pkl"), 'wb')
    pickle.dump(rows, outfile)


