import torch
from scipy import signal
import numpy as np
from numpy.random import RandomState
import pickle
import soundfile as sf
from .hparams import hparams
from SpeechSplit.make_spect_f0 import get_f0
from SpeechSplit.audioRead import get_id
from SpeechSplit.utils import *
from SpeechSplit.model import Generator_3 as Generator
from SpeechSplit.model import Generator_6 as F0_Converter


class SpeechSplitInferencer(object):
	def __init__(self,args):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.hparams = hparams
		self.G_path = args.ss_g
		self.P_path = args.ss_p
		self.G, self.P = self.load_models()

		self.min_level = np.exp(-100 / 20 * np.log(10))
		self.mel_basis = mel(args.sample_rate, hparams.window_length, 
										fmin=90, fmax=7600, n_mels=hparams.cin_channels).T

		self.spk2gen = pickle.load(open('SpeechSplit/assets/spk2gen.pkl', "rb"))
		self.b, self.a = butter_highpass(30, args.sample_rate, order=5)
		
	def load_models(self):
		G = Generator(self.hparams).eval().to(self.device)
		g_checkpoint = torch.load(self.G_path, map_location=lambda storage, loc: storage)
		G.load_state_dict(g_checkpoint['model'])

		P = F0_Converter(self.hparams).eval().to(self.device)
		p_checkpoint = torch.load(self.P_path, map_location=lambda storage, loc: storage)
		P.load_state_dict(p_checkpoint['model'])
		return G, P

	def gen_mel(self, path):
        # read audio file
		audio, sr = sf.read(path)
		assert sr == self.hparams.sample_rate
		if audio.shape[0] % 256 == 0:
			audio = np.concatenate((audio, np.array([1e-06])), axis=0)
		y = signal.filtfilt(self.b, self.a, audio)
		wav = y * 0.96 + (self.prng.rand(y.shape[0])-0.5)*1e-06

		# compute spectrogram
		D = pySTFT(wav).T
		D_mel = np.dot(D, self.mel_basis)
		D_db = 20 * np.log10(np.maximum(self.min_level, D_mel)) - 16
		mel = (D_db + 100) / 100    

		return wav, mel, sr

	def read_audio(self, src_path, trg_path):
		self.trg_spkr = trg_path.split('/')[-1].split('_')[0]
		if self.spk2gen[self.trg_spkr] == 'M':
			lo, hi = 50, 250
		elif self.spk2gen[self.trg_spkr] == 'F':
			lo, hi = 100, 600
		else:
			raise ValueError("Speaker not in dataset")
		print(f"Found target speaker {self.trg_spkr}, loading features...")
		self.prng = RandomState(int(self.trg_spkr[1:])) 

		# get src and trg mels
		_,       src_mel, src_sr = self.gen_mel(src_path)
		trg_wav, trg_mel, trg_sr = self.gen_mel(trg_path)
		assert src_sr == trg_sr

		# get trg f0
		_, trg_f0_norm = get_f0(trg_wav, lo, hi, trg_sr)

		return src_mel, trg_mel, trg_f0_norm


	def pad_utt(self, utterance, max_len=192):
		utt_pad = pad_seq_to_2(utterance[np.newaxis,:,:], max_len)
		utt_pad = torch.from_numpy(utt_pad).to(self.device)
		return utt_pad

	def prep_data(self, src_mel, trg_mel, trg_f0_norm):
		max_len = self.hparams.max_len_pad
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
		src_utt = src_utt.type(torch.float32)
		trg_utt, trg_f0 = trg_utt.type(torch.float32), trg_f0.type(torch.float32)
		# P forward
		with torch.no_grad():
			f0_pred = self.P(src_utt, trg_f0)[0]
			f0_pred_quantized = f0_pred.argmax(dim=-1).squeeze(0)
			f0_con_onehot = torch.zeros((1, self.hparams.max_len_pad, 257), device=self.device)
			f0_con_onehot[0, torch.arange(self.hparams.max_len_pad), f0_pred_quantized] = 1
		uttr_f0_trg = torch.cat((src_utt, f0_con_onehot), dim=-1)    
		# G forward
		emb = get_id(self.trg_spkr, self.spk2gen)
		emb = torch.tensor(emb).unsqueeze(0).to(self.device)
		with torch.no_grad():
			utt_pred = self.G(uttr_f0_trg, trg_utt, emb)
			utt_pred = utt_pred[0,:uttr_f0_trg.shape[1],:].cpu().numpy()
		return utt_pred