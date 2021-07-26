import os
import pickle
import numpy as np
import json
import torch
from scipy import signal
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass, speaker_normalization
from hparams import hparams
from audioRead import MAX_WAV_VALUE
from Waveglow.mel2samp import load_wav_to_torch, Mel2Samp

SAMPLE_RATE = hparams.sample_rate

def get_f0(wav, lo, hi, fs):
    f0_rapt = sptk.rapt(wav.astype(np.float32)*MAX_WAV_VALUE, fs, hparams.hop_size, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
    return f0_rapt, f0_norm

def make_train_data(rootDir, targetDir_f0, targetDir, waveglow_config): 

    MelProcessor = Mel2Samp(**waveglow_config)

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
            audio, fs = load_wav_to_torch(os.path.join(dirName,subdir, fileName))
            assert fs == SAMPLE_RATE

            if audio.shape[0] % hparams.hop_size == 0:
                audio = torch.tensor(np.concatenate((audio, np.array([1e-06])), axis=0), dtype=torch.float32)
            
            # compute spectrogram
            melspectrogram = MelProcessor.get_mel(audio).T.numpy()
            # extract f0
            y = signal.filtfilt(b, a, audio)
            wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
            _, f0_norm = get_f0(wav, lo, hi, fs)
            
            assert len(melspectrogram) == len(f0_norm)
            print(f"Saving {fileName}")
            np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                    melspectrogram.astype(np.float32), allow_pickle=False)    
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


