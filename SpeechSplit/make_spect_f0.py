import os
import pickle
import numpy as np
import json
from pathlib import Path
import soundfile as sf
from scipy import signal
from numpy.random import RandomState
from pysptk import sptk
from utils import *
from hparams import hparams
from audioRead import MAX_WAV_VALUE
from multiprocessing import Process, Manager  

SAMPLE_RATE = hparams.sample_rate
min_level = np.exp(-100 / 20 * np.log(10))
mel_basis = mel(SAMPLE_RATE, hparams.window_length, fmin=90, fmax=7600, n_mels=80).T
b, a = butter_highpass(30, SAMPLE_RATE, order=5)

def get_f0(wav, lo, hi, fs):
    f0_rapt = sptk.rapt(wav.astype(np.float32)*MAX_WAV_VALUE, fs, hparams.hop_size, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
    return f0_rapt, f0_norm

def process_data(dirName, subdir, fileName, mel_target, f0_target, lo, hi, prng, save_f0, save_type):
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

    if save_f0:
        _, f0_norm = get_f0(wav, lo, hi, fs)
        assert len(mel) == len(f0_norm)

    print(f"Saving {fileName}")
    if save_type == 'numpy':
        np.save(os.path.join(mel_target, subdir, fileName[:-4]),
                mel.astype(np.float32), allow_pickle=False)    
        if save_f0:
            np.save(os.path.join(f0_target, subdir, fileName[:-4]),
                    f0_norm.astype(np.float32), allow_pickle=False)
    elif save_type == 'torch':
        torch.save((mel.astype(np.float32),x),
                os.path.join(mel_target, subdir, fileName[:-4] + '.pt'))
        if save_f0:
            torch.save(f0_norm.astype(np.float32), 
                    os.path.join(f0_target, subdir, fileName[:-4] + '.pt'))
    else:
        raise ValueError("Invalid save type")


def make_train_data(rootDir, targetDir,save_f0=False, save_type='numpy'): 

    dirName, subdirList, _ = next(os.walk(rootDir))
    print('Found directory: %s' % dirName)

    # mel(SR, nfft (num fast fourier transform bins))

    spk2gen = pickle.load(open('SpeechSplit/assets/spk2gen.pkl', "rb"))

    for count, subdir in enumerate(sorted(subdirList)):
        if count >= hparams.dim_spk_emb:
            break
        print(f"Processing {subdir}, {count+1} of {hparams.dim_spk_emb}")

        mel_target = targetDir + '/spmel'
        f0_target = targetDir + '/f0'
        if not os.path.exists(os.path.join(mel_target, subdir)):
            os.makedirs(os.path.join(mel_target, subdir))
        if save_f0 and not os.path.exists(os.path.join(f0_target, subdir)):
            os.makedirs(os.path.join(f0_target, subdir))    
        _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
        
        if spk2gen[subdir] == 'M':
            lo, hi = 50, 250
        elif spk2gen[subdir] == 'F':
            lo, hi = 100, 600
        else:
            raise ValueError("Speaker not in dataset")
            
        prng = RandomState(int(subdir[1:])) 
        processes = []
        for fileName in sorted(fileList):
            p = Process(target=process_data, args=(dirName, subdir, fileName, mel_target, f0_target, lo, hi, prng, save_f0, save_type))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        wav_fpaths = list(Path(mel_target).glob(f"**/*.pt"))
        num_wavs = len(wav_fpaths)
        with open(f"{targetDir}/spmel/train_files.txt","w") as f:
            for i, wav_fpath in enumerate(wav_fpaths):
                f.write(str(wav_fpath) + "\n")
                if i > int(num_wavs * 0.9):
                    break
        with open(f"{targetDir}/spmel/test_files.txt","w") as f:
            for i, wav_fpath in enumerate(wav_fpaths):
                if i <= int(num_wavs * 0.9):
                    continue
                f.write(str(wav_fpath) + "\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data_dir", required=True)
    parser.add_argument('-m', "--output_dir", required=True)
    parser.add_argument('-save_type', required=False, default='numpy')
    parser.add_argument('--save_f0', action='store_true')
    args = parser.parse_args()

    make_train_data(args.data_dir, args.output_dir, args.save_f0, args.save_type)


