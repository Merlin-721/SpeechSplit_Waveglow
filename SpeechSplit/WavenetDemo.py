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

# Read in data [original, target]
sbmt_i, sbmt_j = pickle.load(open('assets/demo.pkl', "rb"))
# my_i, my_j = pickle.load(open('processed_assets/customData.pkl', "rb"))

def readRow(sbmt):
    # Get one-hot speaker embedding
    emb = torch.from_numpy(sbmt[1]).to(device)
    # utterance, f0, length of utterance, uid
    x, f0, len_, uid = sbmt[2]        
    # pad utterence and get length of padding
    uttr_pad, len_pad = pad_seq_to_2(x[np.newaxis,:,:], 192)
    uttr_pad = torch.from_numpy(uttr_pad).to(device)
    # pad f0 with same amount
    f0_pad = np.pad(f0, (0, 192-len_), 'constant', constant_values=(0, 0))
    # quantize f0 -> 0-1 into 256 bins, turned to one-hot
    f0_quantized = quantize_f0_numpy(f0_pad)[0]
    # get onehot f0
    f0_onehot = f0_quantized[np.newaxis, :, :]
    f0_onehot = torch.from_numpy(f0_onehot).to(device)
    return emb, len_, uid, uttr_pad, f0_onehot, f0_quantized

# Source/Original
emb_org, len_org, uid_org, uttr_org_pad, f0_org_onehot, f0_org_quantized = readRow(sbmt_i)
uttr_f0_org = torch.cat((uttr_org_pad, f0_org_onehot), dim=-1)

# Target
emb_trg, len_trg, uid_trg, uttr_trg_pad, f0_trg_onehot, f0_trg_quantized = readRow(sbmt_j)

with torch.no_grad():
    f0_pred = P(uttr_org_pad, f0_trg_onehot)[0]
    f0_pred_quantized = f0_pred.argmax(dim=-1).squeeze(0)
    f0_con_onehot = torch.zeros((1, 192, 257), device=device)
    f0_con_onehot[0, torch.arange(192), f0_pred_quantized] = 1
uttr_f0_trg = torch.cat((uttr_org_pad, f0_con_onehot), dim=-1)    

# Rhythm, Frequency, Utterance
# conditions = ['R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU']
conditions = ['RFU']
spect_vc = []
with torch.no_grad():
    for condition in conditions:
        if condition == 'R':
            # Runs forward pass of Generator_3
            x_identic_val = G(uttr_f0_org, uttr_trg_pad, emb_org) # original embedding as rhythm only
        elif condition == 'F':
            x_identic_val = G(uttr_f0_trg, uttr_org_pad, emb_org) # orig emb as freq only
        elif condition == 'U':
            x_identic_val = G(uttr_f0_org, uttr_org_pad, emb_trg) # target emb as utterance
        elif condition == 'RF':
            x_identic_val = G(uttr_f0_trg, uttr_trg_pad, emb_org)
        elif condition == 'RU':
            x_identic_val = G(uttr_f0_org, uttr_trg_pad, emb_trg)
        elif condition == 'FU':
            x_identic_val = G(uttr_f0_trg, uttr_org_pad, emb_trg)
        elif condition == 'RFU':
            x_identic_val = G(uttr_f0_trg, uttr_trg_pad, emb_trg)
            
        if 'R' in condition:
            uttr_trg = x_identic_val[0, :len_trg, :].cpu().numpy()
        else:
            uttr_trg = x_identic_val[0, :len_org, :].cpu().numpy()
                
        spect_vc.append( ('{}_{}_{}_{}'.format(sbmt_i[0], sbmt_j[0], uid_org, condition), uttr_trg ) )       


# Write spectrogram for use elsewhere
# outfile = open("spectrograms.pkl", 'wb')
# pickle.dump(spect_vc, outfile)

# WAVEGEN BITS
import soundfile
import os
from synthesis import build_model
from synthesis import wavegen

if not os.path.exists('results'):
    os.makedirs('results')

model = build_model().to(device)
checkpoint = torch.load("assets/checkpoint_step001000000_ema.pth", map_location=torch.device(device))
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)   
    soundfile.write('results/'+name+'.wav', waveform, samplerate=16000)