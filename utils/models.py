import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence

class ConvCritic(nn.Module):
    def __init__(self, input_size=500, leads_n=12, label_size=7, hidden_size=256, device='cpu'):
        super(ConvCritic, self).__init__()
        self.input_size = input_size
        self.leads_n = leads_n
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.device = device
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.leads_n, 
                      out_channels=self.leads_n * 4, 
                      kernel_size=7, 
                      stride=2, 
                      padding=3, 
                      dilation=2, 
                      groups=self.leads_n),
            nn.BatchNorm1d(self.leads_n * 4, track_running_stats=False),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=self.leads_n * 4, 
                      out_channels=self.leads_n * 4, 
                      kernel_size=7, 
                      stride=2, 
                      padding=3, 
                      dilation=2, 
                      groups=self.leads_n),
            nn.BatchNorm1d(self.leads_n * 4, track_running_stats=False),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=self.leads_n * 4, 
                      out_channels=self.leads_n * 6, 
                      kernel_size=7, 
                      stride=2, 
                      padding=3, 
                      dilation=2, 
                      groups=self.leads_n),
            nn.AvgPool1d(kernel_size=2),          
            ).to(self.device)
        first_l_output = int((self.input_size-7)/2 + 1)//2
        second_l_output = int((first_l_output-7)//2 + 1)//2
        third_l_output = int((second_l_output-7)//2 + 1)//2
        conf_out_size = int(self.leads_n * 6 * third_l_output)
        
        self.fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(conf_out_size, self.hidden_size)
        ).to(self.device)
        
        self.label_encoder = nn.Linear(self.label_size, self.hidden_size).to(self.device)
        
        self.out = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size * 2, 1)
        ).to(self.device)
        
    def forward(self, sequence, label):
        after_c = self.conv(sequence.transpose(2,1))
        after_c = self.fc(after_c.view(after_c.size(0), -1))
        after_encoder = self.label_encoder(label)
        x = torch.cat([after_c, after_encoder], dim=1)
        return torch.sigmoid(self.out(x))
        
        
class InverseCNNGenerator(nn.Module):
    def __init__(self, 
                 input_size, 
                 label_size, 
                 lead_n, 
                 hidden_size, 
                 stride=1, 
                 padding=2, 
                 dilation=2, 
                 kernel_size=7, 
                 out_padding=2,
                 device='cpu'
                ):
        super(InverseCNNGenerator, self).__init__()
        self.input_size = input_size
        self.label_size = label_size
        self.lead_n = lead_n
        self.hidden_size = hidden_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.out_padding = out_padding
        self.device = device
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, 
                               out_channels=self.lead_n * 2,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding,
                               output_padding=self.out_padding,
                               groups=1,
                               dilation=self.dilation,
                              ),
            nn.BatchNorm1d(self.leads_n * 2, track_running_stats=False),
            nn.PRReLU(),
            nn.ConvTranspose1d(in_channels=self.lead_n * 2, 
                               out_channels=self.lead_n * 4,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding,
                               output_padding=self.out_padding,
                               groups=self.lead_n,
                               dilation=self.dilation,
                              ),
            nn.BatchNorm1d(self.leads_n * 2, track_running_stats=False),
            nn.PRReLU(),
            nn.ConvTranspose1d(in_channels=self.lead_n * 4, 
                               out_channels=self.lead_n * 4,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding,
                               output_padding=self.out_padding,
                               groups=self.lead_n,
                               dilation=self.dilation,
                              ),
            nn.BatchNorm1d(self.leads_n * 2, track_running_stats=False),
            nn.PRReLU(),
            nn.ConvTranspose1d(in_channels=self.lead_n * 4, 
                               out_channels=self.lead_n * 2,
                               kernel_size=4,
                               stride=1,
                               padding=3,
                               groups=self.lead_n
                              ),
            nn.BatchNorm1d(self.leads_n * 2, track_running_stats=False),
            nn.PRReLU(),
            nn.ConvTranspose1d(in_channels=self.lead_n * 2, 
                               out_channels=self.lead_n * 2,
                               kernel_size=4,
                               stride=1,
                               padding=3,
                               groups=self.lead_n
                              ),
            nn.PRReLU(),
            nn.ConvTranspose1d(in_channels=self.lead_n * 2, 
                               out_channels=self.lead_n,
                               kernel_size=4,
                               stride=2,
                               padding=2,
                               groups=self.lead_n
                              )
        ).to(self.device)
        
        self.encoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=self.label_size + self.input_size, out_features=self.hidden_size)
        ).to(self.device)
        
    def forward(self, noise, label):
        x = self.encoder(torch.cat([noise, label], dim=1)).unsqueeze(1)     
        return torch.tanh(self.deconv(x)).transpose(1,2)


class DenseGenerator(nn.Module):
    def __init__(self, noise_size, label_size, output_size, hidden_size, n_lead, device='cpu'):
        super(DenseGenerator, self).__init__()
        self.noise_size = noise_size
        self.label_size = label_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_lead = n_lead
        self.device = device
        
        self.noise_encoder = nn.Sequential(
                nn.Linear(self.noise_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size, track_running_stats=False),
                nn.PRReLU()
        ).to(self.device)
        self.label_encoder = nn.Sequential(
                nn.Linear(self.label_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size, track_running_stats=False),
                nn.PRReLU()
        ).to(self.device)
        self.lead_generators = [nn.Linear(self.hidden_size*2, self.output_size, bias=False).to(self.device) for _ in range(self.n_lead)]
        
    def forward(self, noise, label):
        noise = self.noise_encoder(noise)
        label = self.label_encoder(label)
        x = torch.cat([noise, label], dim=1)
        generated_lead_list = [torch.tanh(lead_g(x)).unsqueeze(-1) for lead_g in self.lead_generators]
        return torch.cat(generated_lead_list, dim=2)        
    
class DenseCritic(nn.Module):
    def __init__(self, input_size, label_size, hidden_size, lead_n, device='cpu'):
        super(DenseCritic, self).__init__()
        self.input_size = input_size
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.lead_n = lead_n
        self.device = device

        self.label_encoder = nn.Sequential(
            nn.Linear(self.label_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2, track_running_stats=False),
            nn.PRReLU()
        ).to(self.device)
        self.lead_encoder_list = [
            nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size//2),
                nn.BatchNorm1d(self.hidden_size//2, track_running_stats=False),
                nn.PRReLU()
            ).to(self.device) for _ in range(self.lead_n)]
        self.critic = nn.Linear(self.hidden_size, 1).to(self.device)
        
    def forward(self, sequence, label):
        encoded_leads = [lead_encoder(sequence[:,:,i]) for i, lead_encoder in enumerate(self.lead_encoder_list)]
        encoded_label = self.label_encoder(label)
        critic_outputs = [torch.sigmoid(self.critic(torch.cat([lead, encoded_label], dim=1))) for lead in encoded_leads]
        return torch.cat(critic_outputs, dim=1)
        

class LSTMGenerator(nn.Module):
    def __init__(self, labels_dim, output_dim, hidden_dim, noise_size, n_lead=12, device='cpu'):
        super(Generator, self).__init__()
        self.labels_dim = labels_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.noise_size = noise_size
        self.device = device
        self.n_lead = n_lead
        self.lstm_output_mult = 2
        
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=self.hidden_dim,
                            num_layers=2,
                            batch_first=True,
                            dropout=0.4,
                            bidirectional=True,
                            ).to(self.device)
        self.label_encoder = nn.Sequential(
            nn.Linear(self.labels_dim, self.hidden_dim)
        )
        self.out = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * self.lstm_output_mult + self.hidden_dim, track_running_stats=False),
            nn.PRReLU(),
            nn.Linear(self.hidden_dim * self.lstm_output_mult + self.hidden_dim, self.n_lead, bias=False)
        ).to(self.device)

    def init_hidden(self, batch_size):
        return tuple([torch.zeros((2, batch_size, self.hidden_dim), dtype=torch.float32).to(self.device)
                      for _ in range(2)])

    def forward(self, noise, labels):
        encoded_seq, hidden = self.lstm(noise, self.init_hidden(noise.size(0))
        encoded_labels = self.label_encoder(labels)
        x = torch.cat([encoded_seq, encoded_labels.expand(noise.size(1), -1)], dim=1)
        return torch.tanh(self.out(x))


class LSTMCritic(nn.Module):
    def __init__(self, input_size, label_size, hidden_size, lead_n, device='cpu'):

        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.lead_n = lead_n
        self.decoder_dim = decoder_dim
        self.device = device

        self.lstm = nn.LSTM(input_size=self.lead_n,
                            hidden_size=self.hidden_dim,
                            num_layers=2,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=True
                            ).to(self.device)
                                        
        self.out = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * self.input_size, track_running_stats=False),
            nn.AvgPool1d(kernel_size=2),
            nn.PRReLU(),
            nn.Linear(self.hidden_dim * self.input_size // 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim, track_running_stats=False),
            nn.PRReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)

    def init_hidden(self, batch_size):
        return tuple([torch.zeros((2, batch_size, self.hidden_dim), dtype=torch.float32).to(self.device) for _ in range(2)])

    def forward(self, seqs, labels):
        x, hidden = self.lstm(seqs)
        return self.out(x.view(x.size(0), -1))
