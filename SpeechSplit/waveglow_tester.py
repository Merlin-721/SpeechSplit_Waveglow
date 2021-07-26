import torch
import pickle
import numpy as np
from hparams import hparams
from utils import pad_seq_to_2
from utils import quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(str(device))

G = Generator(hparams).eval().to(device)
g_checkpoint = torch.load('assets/660000-G.ckpt', map_location=lambda storage, loc: storage)
G.load_state_dict(g_checkpoint['model'])

P = F0_Converter(hparams).eval().to(device)
p_checkpoint = torch.load('assets/640000-P.ckpt', map_location=lambda storage, loc: storage)
P.load_state_dict(p_checkpoint['model'])

def makeOnehot(element, length, items):
	location = items.index(element)
	onehot = np.zeros(length,dtype=np.float32)
	onehot[location] = 1
	return torch.tensor(onehot)
	
src = "p226"
trg = "p228"
speaker_ids = pickle.load(open("assets/spk2gen.pkl", "rb"))
spkr_list = list(speaker_ids.keys())
spkr_list = sorted(spkr_list, key=lambda item: (int(item.partition(' ')[0])
                               if item[0].isdigit() else float('inf'), item))
source_embedding = makeOnehot(src, hparams.dim_spk_emb, spkr_list).unsqueeze(0).to(device)
target_embedding = makeOnehot(trg, hparams.dim_spk_emb, spkr_list).unsqueeze(0).to(device)

# source_embedding = torch.from_numpy(speaker_ids[src]).to(device)
# target_embedding = torch.from_numpy(speaker_ids[trg]).to(device)
source_mel = np.load("assets/spmel/p226/p226_003.npy")
source_f0 = np.load("assets/raptf0/p226/p226_003.npy")
target_mel = np.load("assets/spmel/p228/p228_024.npy")
target_f0 = np.load("assets/raptf0/p228/p228_024.npy")

def readData(mel, f0):
	# pad utterence and get length of padding
	print(mel.shape)
	uttr_pad = mel[np.newaxis,:192,:]
	# uttr_pad, len_pad = pad_seq_to_2(mel[np.newaxis,:,:], 192)
	uttr_pad = torch.from_numpy(uttr_pad).to(device)
	# pad f0 with same amount
	f0_pad = f0[:192]
	# quantize f0 -> 0-1 into 256 bins, turned to one-hot
	f0_quantized = quantize_f0_numpy(f0_pad)[0]
	# get onehot f0
	f0_onehot = f0_quantized[np.newaxis, :, :]
	f0_onehot = torch.from_numpy(f0_onehot).to(device)
	return uttr_pad, f0_onehot, f0_quantized

# Source/Original
uttr_org_pad, f0_org_onehot, f0_org_quantized = readData(source_mel, source_f0)
uttr_f0_org = torch.cat((uttr_org_pad, f0_org_onehot), dim=-1)

# Target
uttr_trg_pad, f0_trg_onehot, f0_trg_quantized = readData(target_mel, target_f0)


with torch.no_grad():
    f0_pred = P(uttr_org_pad, f0_trg_onehot)[0]
    f0_pred_quantized = f0_pred.argmax(dim=-1).squeeze(0)
    f0_con_onehot = torch.zeros((1, 192, 257), device=device)
    f0_con_onehot[0, torch.arange(192), f0_pred_quantized] = 1
uttr_f0_trg = torch.cat((uttr_org_pad, f0_con_onehot), dim=-1)    

# Rhythm, Frequency, Utterance
condition = 'RFU'
# spect_vc = []
with torch.no_grad():
	x_identic_val = G(uttr_f0_trg, uttr_trg_pad, target_embedding)
	uttr_trg = x_identic_val[0, :target_mel.shape[0], :].cpu().numpy()
	torch.save(uttr_trg, "output_mels/p226_p228_tester.wav.pt")
        # spect_vc.append( ('{}_{}_{}_{}'.format(sbmt_i[0], sbmt_j[0], uid_org, condition), uttr_trg ) )       

print()