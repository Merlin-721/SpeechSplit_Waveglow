import torch
from scipy import signal
import numpy as np
import pickle
import json
from types import SimpleNamespace
from pathlib import Path

from SpeechSplit.make_spect_f0 import get_f0
from SpeechSplit.audioRead import get_id
from SpeechSplit.utils import *
from SpeechSplit.model import Generator_3 as Generator
from SpeechSplit.model import Generator_6 as F0_Converter
from Waveglow.mel2samp import load_wav_to_torch, Mel2Samp

class SpeechSplitInferencer(object):
	def __init__(self,args, waveglow_config):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		with open(args.speech_split_conf) as f:
			self.config = json.load(f,object_hook=lambda d: SimpleNamespace(**d))
		self.G_path = args.ss_g
		self.G = self.load_models()
		self.MelProcessor = Mel2Samp(**waveglow_config)

		self.spk2gen = pickle.load(open('SpeechSplit/assets/spk2gen.pkl', "rb"))
		self.b, self.a = butter_highpass(30, args.sample_rate, order=5)
		
	def load_models(self):
		G = Generator(self.config).eval().to(self.device)
		g_checkpoint = torch.load(self.G_path, map_location=lambda storage, loc: storage)
		G.load_state_dict(g_checkpoint['model'])

		return G

	def gen_mel(self, path):
		audio, sr = load_wav_to_torch(path)
		if audio.shape[0] % self.config.hop_size == 0:
			audio = torch.tensor(np.concatenate((audio, np.array([1e-06])), axis=0), dtype=torch.float32)
		mel = self.MelProcessor.get_mel(audio).T.numpy()
		return audio, mel, sr
	
	def gen_f0(self, audio, lo, hi, sr):
		y = signal.filtfilt(self.b, self.a, audio)
		wav = y * 0.96 + np.random.rand(y.shape[0])*1e-06
		_, f0 = get_f0(wav, lo, hi, sr)
		return f0

	def read_audio(self, src_path, trg):
		self.trg_spkr = trg
		if self.spk2gen[self.trg_spkr] == 'M':
			lo, hi = 50, 250
		elif self.spk2gen[self.trg_spkr] == 'F':
			lo, hi = 100, 600
		else:
			raise ValueError("Speaker not in dataset")
		print(f"Found target speaker {self.trg_spkr}, loading features...")

		# get src and trg mels
		src_audio, src_mel, src_sr = self.gen_mel(src_path)

		# get trg f0
		src_f0_norm = self.gen_f0(src_audio, lo, hi, src_sr)

		assert src_mel.shape[0] == src_f0_norm.shape[0]

		return src_mel, src_f0_norm


	def pad_utt(self, utterance, max_len=192):
		utt_pad = pad_seq_to_2(utterance[np.newaxis,:,:], max_len)
		utt_pad = torch.from_numpy(utt_pad).to(self.device)
		return utt_pad

	def prep_data(self, src_mel, src_f0_norm, pad=True):
		if pad:
			max_len = self.config.max_len_pad
			src_utt_pad = self.pad_utt(src_mel, max_len)
			src_f0_pad = pad_f0(src_f0_norm.squeeze(), max_len)
		else:
			# max_len = src_mel.shape[0] // 64 * 64
			max_len = src_mel.shape[0] // self.config.freq * self.config.freq
			src_utt_pad = self.pad_utt(src_mel, max_len)
			src_f0_pad = pad_f0(src_f0_norm.squeeze(), max_len)
		src_f0_quant = quantize_f0_numpy(src_f0_pad)[0]
		src_f0_onehot = src_f0_quant[np.newaxis,:,:]
		src_f0_onehot = torch.from_numpy(src_f0_onehot).to(self.device)
		
		return src_utt_pad, src_f0_onehot

	def forward(self, src_utt, src_f0):
		# src utt is padded
		# src f0 is onehot
		emb = get_id(self.trg_spkr, self.spk2gen)
		emb = torch.tensor(emb).unsqueeze(0).to(self.device)

		uttr_f0_org = torch.cat((src_utt, src_f0), dim=-1)    
		# G forward
		with torch.no_grad():
			utt_pred = self.G(uttr_f0_org, src_utt, emb)
			utt_pred = utt_pred[0,:uttr_f0_org.shape[1],:].cpu().numpy()
		return utt_pred