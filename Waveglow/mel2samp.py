# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
import numpy as np
from numpy.random import RandomState
from librosa.filters import mel
from scipy.io.wavfile import read
import soundfile as sf
from scipy import signal
# from SpeechSplit.utils import butter_highpass, pySTFT

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
# from SpeechSplit.tacotron2.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0


def butter_highpass(cutoff, fs, order=5):
    # Returns b (numerator) and a (denominator) polynomials
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = signal.get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        try: 
            self.audio_files = files_to_list(training_files)
            random.seed(1234)
            random.shuffle(self.audio_files)
        except:
            pass
        # self.stft = TacotronSTFT(filter_length=filter_length,
        #                          hop_length=hop_length,
        #                          win_length=win_length,
        #                          sampling_rate=sampling_rate,
        #                          mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

        self.min_level = np.exp(-100 / 20 * np.log(10))
        self.mel_basis = mel(sampling_rate, win_length, fmin=90, fmax=7600, n_mels=80).T
        self.b, self.a = butter_highpass(30, sampling_rate, order=5)

    def get_mel(self, audio):
        if audio.shape[0] % 256 == 0:
            audio = np.concatenate((audio, np.array([1e-06])), axis=0)
        y = signal.filtfilt(self.b, self.a, audio)
        prng = RandomState(1)
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        
        # compute spectrogram
        D = pySTFT(wav).T
        D_mel = np.dot(D, self.mel_basis)
        D_db = 20 * np.log10(np.maximum(self.min_level, D_mel)) - 16
        mel = torch.tensor((D_db + 100) / 100 )
        return mel.T

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio = sf.read(filename)[0]
        audio = torch.tensor(audio)
        # audio = torch.load(filename)
        # audio, sampling_rate = load_wav_to_torch(filename)
        # if sampling_rate != self.sampling_rate:
        #     raise ValueError("{} SR doesn't match target {} SR".format(
        #         sampling_rate, self.sampling_rate))

        # Take segment
        if audio.shape[0] >= self.segment_length:
            max_audio_start = audio.shape[0] - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        mel = self.get_mel(audio)
        mel = mel.type(torch.float32)
        audio = audio.type(torch.float32)
        # audio = audio / MAX_WAV_VALUE

        return (mel, audio)

    def __len__(self):
        return len(self.audio_files)

# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    mel2samp = Mel2Samp(**data_config)

    filepaths = files_to_list(args.filelist_path)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    for filepath in filepaths:
        audio, sr = load_wav_to_torch(filepath)
        melspectrogram = mel2samp.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)
