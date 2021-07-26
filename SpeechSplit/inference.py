import torch
from scipy import signal
import numpy as np
import pickle

from SpeechSplit.make_spect_f0 import get_f0
from SpeechSplit.audioRead import get_id
from SpeechSplit.utils import *
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

	def gen_mel(self, path):
		audio, sr = load_wav_to_torch(path)
		if audio.shape[0] % self.config.hop_size == 0:
			audio = torch.tensor(np.concatenate((audio, np.array([1e-06])), axis=0), dtype=torch.float32)
		mel = self.MelProcessor.get_mel(audio).T.numpy()
		return audio, mel, sr

	def read_audio(self, src_path, trg_path):
		self.trg_spkr = trg_path.split('/')[-1].split('_')[0]
		if self.spk2gen[self.trg_spkr] == 'M':
			lo, hi = 50, 250
		elif self.spk2gen[self.trg_spkr] == 'F':
			lo, hi = 100, 600
		else:
			raise ValueError("Speaker not in dataset")
		print(f"Found target speaker {self.trg_spkr}, loading features...")

		# get src and trg mels
		_        , src_mel, src_sr = self.gen_mel(src_path)
		trg_audio, trg_mel, trg_sr = self.gen_mel(src_path)
		assert src_sr == trg_sr

		# get trg f0
		y = signal.filtfilt(self.b, self.a, trg_audio)
		wav = y * 0.96 + np.random.rand(y.shape[0])*1e-06
		_, trg_f0_norm = get_f0(wav, lo, hi, trg_sr)

		return src_mel, trg_mel, trg_f0_norm


	def pad_utt(self, utterance, max_len=192):
		utt_pad, _ = pad_seq_to_2(utterance[np.newaxis,:,:], max_len)
		utt_pad = torch.from_numpy(utt_pad).to(self.device)
		return utt_pad

	def prep_data(self, src_mel, trg_mel, trg_f0_norm):
		max_len = self.config.max_len_pad
		src_utt_pad = self.pad_utt(src_mel, max_len)
		trg_utt_pad = self.pad_utt(trg_mel, max_len)

		trg_f0_pad = pad_f0(trg_f0_norm.squeeze(), max_len)
		trg_f0_quant = quantize_f0_numpy(trg_f0_pad)[0]
		trg_f0_onehot = trg_f0_quant[np.newaxis,:,:]
		trg_f0_onehot = torch.from_numpy(trg_f0_onehot).to(self.device)
		
		return src_utt_pad, trg_utt_pad, trg_f0_onehot

	def forward(self, src_utt, trg_utt, trg_f0):
		# src utt is padded
		# trg f0 is onehot
		emb = get_id(self.trg_spkr, self.spk2gen)
		# P forward
		with torch.no_grad():
			f0_pred = self.P(src_utt, trg_f0)[0]
			f0_pred_quantized = f0_pred.argmax(dim=-1).squeeze(0)
			f0_con_onehot = torch.zeros((1, self.config.max_len_pad, 257), device=self.device)
			f0_con_onehot[0, torch.arange(self.config.max_len_pad), f0_pred_quantized] = 1
		uttr_f0_trg = torch.cat((src_utt, f0_con_onehot), dim=-1)    
		# G forward
		with torch.no_grad():
			utt_pred = self.G(uttr_f0_trg, trg_utt, emb)
			utt_pred = utt_pred[0,:uttr_f0_trg.shape[0],:].cpu().numpy()
		return utt_pred