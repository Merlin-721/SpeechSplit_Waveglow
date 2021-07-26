import torch
from utils import pad_seq_to_2
from utils import quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter


class SpeechSplitInferencer(object):
	def __init__(self, config):
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.G, self.P = self.load_models()
		
	def load_models(self):
		G = Generator(self.config).eval().to(self.device)
		P = F0_Converter(self.config).eval().to(self.device)
		return G, P