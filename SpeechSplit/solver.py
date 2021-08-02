from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter
from model import InterpLnr
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle
import random
import json

from utils import pad_seq_to_2, pad_f0, quantize_f0_torch, quantize_f0_numpy

# use demo data for simplicity
# make your own validation set as needed
# validation_pt = pickle.load(open('assets/spmel/demo.pkl', "rb"))

class Solver(object):
    """Solver for training"""

    def __init__(self, vcc_loader, config, hparams):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.hparams = hparams
        self.val_root = config.val_path.split('/')[:-1]
        self.val_meta = pickle.load(open(config.val_path, "rb"))

        # Training configurations.
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.p_lr = config.p_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        
        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.save_dir = config.save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        
        self.train_G = config.train_G
        self.train_P = config.train_P

        # Build the model and tensorboard.
        if config.train_G:
            self.build_G()
        if config.train_P:
            self.build_P()
        self.Interp = InterpLnr(self.hparams)
        self.Interp.to(self.device)

        self.dump_hparams()

        if self.use_tensorboard:
            self.build_tensorboard()

    def build_G(self):
        self.G = Generator(self.hparams)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.G.to(self.device)
            
    def build_P(self):
        self.P = F0_Converter(self.hparams)
        self.p_optimizer = torch.optim.Adam(self.P.parameters(), self.p_lr, [self.beta1, self.beta2])
        self.print_network(self.P, 'P')
        self.P.to(self.device)

    def dump_hparams(self):
        with open(f'{self.save_dir}/hparams.json', 'w') as fp:
            json.dump(self.hparams.to_json(indent=0), fp)
        
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(f"\nModel {name}:\n")
        print(model)
        print("The number of parameters: {}".format(num_params))
        
        

    # TODO ADD FOR P 
    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.G_save_dir, '{}-G.ckpt'.format(resume_iters))
        g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
        self.G.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']
        P_path = os.path.join(self.P_save_dir, '{}-P.ckpt'.format(resume_iters))
        p_checkpoint = torch.load(P_path, map_location=lambda storage, loc: storage)
        self.P.load_state_dict(p_checkpoint['model'])
        self.p_optimizer.load_state_dict(p_checkpoint['optimizer'])
        self.p_lr = self.p_optimizer.param_groups[0]['lr']
        
        
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)
        

    def data_to_device(self, args):
        return [x.to(self.device) for x in args]

    def load_validation_data(self):
        self.val_set = []
        for speaker in self.val_meta:
            emb = torch.from_numpy(speaker[1])
            utts, f0s = [], []
            files = speaker[2:]
            for file in files:
                utts.append(np.load(str('/'.join(self.val_root)+'/spmel/'+file)))
                f0s.append(np.load(str('/'.join(self.val_root)+'/raptf0/'+file)))
            self.val_set.append([speaker[0], emb, utts, f0s])


#=====================================================================================================================
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        self.load_validation_data()
        
        
        # Start training from scratch or resume training.
        # TODO ADD P
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            print(f'G Optimiser: \n {self.g_optimizer}')
            print(f'P Optimiser: \n {self.p_optimizer}')
                        
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        p_lr = self.p_lr
        print (f'Current learning rates, G: {g_lr}, P: {p_lr}')
        
            
        # Start training.
        print('Start training...')
        start_time = time.time()

        data_iter = iter(data_loader)
        for i in range(start_iters, self.num_iters):

            # Logging.
            loss = {}
            # =================================================================================== #
            #                               2. Generator Training                                 #
            # =================================================================================== #

            x_real_org, emb_org, f0_org, len_org = next(data_iter)
            x_real_org, emb_org, len_org, f0_org = self.data_to_device([x_real_org, emb_org, len_org, f0_org])

            # combines spect and f0s
            x_f0 = torch.cat((x_real_org, f0_org), dim=-1)
            # Random resampling with linear interpolation
            x_f0_intrp = self.Interp(x_f0, len_org) 
            # strips f0 from trimmed to quantize it
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:,:,-1])[0]

            if self.train_G:
                self.G = self.G.train()
                # combines quantized f0 back with spect
                x_f0_intrp_org = torch.cat((x_f0_intrp[:,:,:-1], f0_org_intrp), dim=-1)

                # G forward
                x_pred = self.G(x_f0_intrp_org, x_real_org, emb_org)
                g_loss_id = F.mse_loss(x_pred, x_real_org, reduction='mean') 

                # Backward and optimize.
                self.g_optimizer.zero_grad()
                g_loss_id.backward()
                self.g_optimizer.step()

                loss['G/loss_id'] = g_loss_id.item()

            # =================================================================================== #
            #                               3. F0_Converter Training                              #
            # =================================================================================== #
            if self.train_P:

                self.P = self.P.train()
                f0_trg_intrp_indx = f0_org_intrp.argmax(2)

                # P forward
                f0_pred = self.P(x_real_org,f0_org_intrp)
                p_loss_id = F.cross_entropy(f0_pred.transpose(1,2),f0_trg_intrp_indx, reduction='mean')


                self.p_optimizer.zero_grad()
                p_loss_id.backward()
                self.p_optimizer.step()
                loss['P/loss_id'] = p_loss_id.item()
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.

            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et_str = str(datetime.timedelta(seconds=et))[:-7]
                seconds = (et / (i+1)) * (self.num_iters - i) 
                mins = seconds / 60
                hrs = mins / 60
                lossString = '; '.join([f'{k} {round(v,8)}' for k, v in loss.items()])
                print( f'Iteration [{i+1}/{self.num_iters}]' + lossString)
                print(f'Time elapsed: {et_str}; Time remaining:{round(hrs/24,2)} days, {round(hrs,2)} hrs or {round(mins,1)} mins')
                for tag, value in loss.items():
                    self.writer.add_scalar(tag, value, i+1)
                        
                        
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0 and i >= int(self.num_iters*0.25):
                print()
                if self.train_G:
                    G_path = os.path.join(self.save_dir, 'G/{}-G.ckpt'.format(i+1))
                    torch.save({'model': self.G.state_dict(),
                                'optimizer': self.g_optimizer.state_dict()}, G_path)
                    print(f"Saved G checkpoint to:\n{G_path}")

                if self.train_P:
                    P_path = os.path.join(self.save_dir, 'P/{}-P.ckpt'.format(i+1))
                    torch.save({'model': self.P.state_dict(),
                                'optimizer': self.p_optimizer.state_dict()}, P_path)
                    print(f"Saved P checkpoint to:\n{P_path}")

            # Validation.
            if (i+1) % self.sample_step == 0:
                self.validate_and_plot(i+1)

#=====================================================================================================================

    def validate_and_plot(self,step):
        self.G = self.G.eval() if self.train_G else None
        self.P = self.P.eval() if self.train_P else None
        with torch.no_grad():
            G_loss_values, P_loss_values = [], []
            random.shuffle(self.val_set)
            for speaker in self.val_set[:10]:
                name, emb, utts, f0s = speaker
                emb = emb.unsqueeze(0).to(self.device)
                k=0
                for utt, f0 in zip(utts, f0s):
                    k+=1
                    x_real_pad = pad_seq_to_2(utt[np.newaxis,:,:], 192)
                    f0_org = pad_f0(f0.squeeze(),192)
                    f0_quantized = quantize_f0_numpy(f0_org)[0]
                    f0_onehot = f0_quantized[np.newaxis, :, :]
                    f0_org_val = torch.from_numpy(f0_onehot).to(self.device) 
                    x_real_pad = torch.from_numpy(x_real_pad).to(self.device) 
                    
                    if self.train_P:
                        # P forward and loss calc
                        f0_pred = self.P(x_real_pad,f0_org_val)
                        f0_trg_intrp_indx = f0_org_val.argmax(2)
                        p_loss_id = F.cross_entropy(f0_pred.transpose(1,2),f0_trg_intrp_indx, reduction='mean')
                        P_loss_values.append(p_loss_id.item())

                    if self.train_G:
                        x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
                        # G forward and loss calc
                        x_identic_val = self.G(x_f0, x_real_pad, emb)
                        g_loss_val = F.mse_loss(x_identic_val,x_real_pad, reduction='mean')
                        G_loss_values.append(g_loss_val.item())

                        x_f0_F = torch.cat((x_real_pad, torch.zeros_like(f0_org_val)), dim=-1)
                        x_f0_C = torch.cat((torch.zeros_like(x_real_pad), f0_org_val), dim=-1)
                        
                        x_identic_val = self.G(x_f0, x_real_pad, emb)
                        x_identic_woF = self.G(x_f0_F, x_real_pad, emb)
                        x_identic_woR = self.G(x_f0, torch.zeros_like(x_real_pad), emb)
                        x_identic_woC = self.G(x_f0_C, x_real_pad, emb)
                        
                        melsp_gd_pad = x_real_pad[0].cpu().numpy().T
                        melsp_out = x_identic_val[0].cpu().numpy().T
                        melsp_woF = x_identic_woF[0].cpu().numpy().T
                        melsp_woR = x_identic_woR[0].cpu().numpy().T
                        melsp_woC = x_identic_woC[0].cpu().numpy().T
                        
                        min_value = np.min(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
                        max_value = np.max(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
                        
                        fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, 1, sharex=True)
                        fig.suptitle(f'Step {step} {name}_{k}', fontsize=12, y=1)
                        ax1.imshow(melsp_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
                        ax1.set_title('Original')
                        ax2.imshow(melsp_out, aspect='auto', vmin=min_value, vmax=max_value)
                        ax2.set_title('G straight conversion')
                        ax3.imshow(melsp_woC, aspect='auto', vmin=min_value, vmax=max_value)
                        ax3.set_title('G w/o Content')
                        ax4.imshow(melsp_woR, aspect='auto', vmin=min_value, vmax=max_value)
                        ax4.set_title('G w/o Rythm')
                        ax5.imshow(melsp_woF, aspect='auto', vmin=min_value, vmax=max_value)
                        ax5.set_title('G w/o F0')
                        fig.subplots_adjust(top=0.901,bottom=0.046,left=0.051,right=0.977,hspace=0.4,wspace=0.2)
                        plt.savefig(f'{self.sample_dir}/{step}_{name}_{k}.png', dpi=150)
                        plt.close(fig)

        G_val_loss = np.mean(G_loss_values) if len(G_loss_values) > 0 else 0
        print(f'G Validation loss: {round(G_val_loss,3)}')
        P_val_loss = np.mean(P_loss_values) if len(P_loss_values) > 0 else 0
        print(f'P Validation loss: {round(P_val_loss,3)}')
        if self.use_tensorboard:
            self.writer.add_scalar('G Validation_loss', G_val_loss, step)
            self.writer.add_scalar('P Validation_loss', P_val_loss, step)
