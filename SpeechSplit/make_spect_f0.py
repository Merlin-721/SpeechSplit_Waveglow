import os
import pickle
import numpy as np
import json
import soundfile as sf
from scipy import signal
from numpy.random import RandomState
from pysptk import sptk
from .utils import *
from .hparams import hparams
from .audioRead import MAX_WAV_VALUE

SAMPLE_RATE = hparams.sample_rate
min_level = np.exp(-100 / 20 * np.log(10))
mel_basis = mel(SAMPLE_RATE, hparams.window_length, fmin=90, fmax=7600, n_mels=80).T

def get_f0(wav, lo, hi, fs):
    f0_rapt = sptk.rapt(wav.astype(np.float32)*MAX_WAV_VALUE, fs, hparams.hop_size, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
    return f0_rapt, f0_norm

def make_train_data(rootDir, targetDir_f0, targetDir): 

    dirName, subdirList, _ = next(os.walk(rootDir))
    print('Found directory: %s' % dirName)

    # mel(SR, nfft (num fast fourier transform bins))
    b, a = butter_highpass(30, SAMPLE_RATE, order=5)

    spk2gen = pickle.load(open('assets/spk2gen.pkl', "rb"))

    for count, subdir in enumerate(sorted(subdirList)):
        if count >= hparams.dim_spk_emb:
            break
        print(f"Processing {subdir}, {count+1} of {hparams.dim_spk_emb}")
        
        if not os.path.exists(os.path.join(targetDir, subdir)):
            os.makedirs(os.path.join(targetDir, subdir))
        if not os.path.exists(os.path.join(targetDir_f0, subdir)):
            os.makedirs(os.path.join(targetDir_f0, subdir))    
        _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
        
        if spk2gen[subdir] == 'M':
            lo, hi = 50, 250
        elif spk2gen[subdir] == 'F':
            lo, hi = 100, 600
        else:
            raise ValueError("Speaker not in dataset")
            
        prng = RandomState(int(subdir[1:])) 
        for fileName in sorted(fileList):
            # read audio file
            x, fs = sf.read(os.path.join(dirName,subdir,fileName))
            assert fs == SAMPLE_RATE
            if x.shape[0] % 256 == 0:
                x = np.concatenate((x, np.array([1e-06])), axis=0)
            y = signal.filtfilt(b, a, x)
            wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
            
            # compute spectrogram
            D = pySTFT(wav).T
            D_mel = np.dot(D, mel_basis)
            D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
            mel = (D_db + 100) / 100     

            _, f0_norm = get_f0(wav, lo, hi, fs)
            
            assert len(mel) == len(f0_norm)
            print(f"Saving {fileName}")
            np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                    mel.astype(np.float32), allow_pickle=False)    
            np.save(os.path.join(targetDir_f0, subdir, fileName[:-4]),
                    f0_norm.astype(np.float32), allow_pickle=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data_dir", required=True)
    parser.add_argument('-f0', "--f0_output_dir", required=True)
    parser.add_argument('-m', "--mel_output_dir", required=True)
    parser.add_argument('-w', "--waveglow_config", required=True, default='Waveglow/config.json')
    args = parser.parse_args()

    with open(args.waveglow_config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]

    make_train_data(args.data_dir, args.f0_output_dir, args.mel_output_dir, data_config)


