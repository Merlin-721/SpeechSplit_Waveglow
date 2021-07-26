import torch
from scipy import signal
import numpy as np
import pickle

from SpeechSplit.make_spect_f0 import get_f0
from SpeechSplit.utils import pad_seq_to_2
from SpeechSplit.utils import quantize_f0_numpy, butter_highpass, speaker_normalization
from SpeechSplit.model import Generator_3 as Generator
from SpeechSplit.model import Generator_6 as F0_Converter
from Waveglow.mel2samp import load_wav_to_torch, Mel2Samp


class SpeechSplitInferencer(object):
	def __init__(self, config, waveglow_config):
		self.config = config
		self.waveglow_config = waveglow_config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.G, self.P = self.load_models()
		self.MelProcessor = Mel2Samp(**waveglow_config)

		self.spk2gen = pickle.load(open('assets/spk2gen.pkl', "rb"))
		self.b, self.a = butter_highpass(30, config.sample_rate, order=5)
		
	def load_models(self):
		G = Generator(self.config).eval().to(self.device)
		# TODO replace with arg
		g_checkpoint = torch.load('SpeechSplit/run/models/22k_waveglow/G/180000-G.ckpt', map_location=lambda storage, loc: storage)
		G.load_state_dict(g_checkpoint['model'])

		P = F0_Converter(self.config).eval().to(self.device)
		# TODO replace with arg
		p_checkpoint = torch.load('SpeechSplit/run/models/22k_waveglow/P/180000-P.ckpt', map_location=lambda storage, loc: storage)
		P.load_state_dict(p_checkpoint['model'])
		return G, P

	def read_audio(self,src_path, trg_path):
		trg_spkr = trg_path.split('/')[-1].split('_')[0]
		if self.spk2gen[trg_spkr] == 'M':
			lo, hi = 50, 250
		elif self.spk2gen[trg_spkr] == 'F':
			lo, hi = 100, 600
		else:
			raise ValueError("Speaker not in dataset")
		print(f"Found target speaker {trg_spkr}, loading features...")

		src_audio, src_sr = load_wav_to_torch(src_path)
		trg_audio, trg_sr = load_wav_to_torch(trg_path)
		assert src_sr == trg_sr

		if src_audio.shape[0] % self.config.hop_size == 0:
			src_audio = torch.tensor(np.concatenate((src_audio, np.array([1e-06])), axis=0), dtype=torch.float32)

		# get src mel
		mel = self.MelProcessor.get_mel(src_audio).T.numpy()

		# get trg f0
		y = signal.filtfilt(self.b, self.a, trg_audio)
		wav = y * 0.96 + np.random.rand(y.shape[0])*1e-06
		_, f0_norm = get_f0(wav, lo, hi, trg_sr)

		return mel, f0_norm
