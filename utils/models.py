import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence

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
                nn.LeakyReLU()).to(self.device)
        self.label_encoder = nn.Sequential(
                nn.Linear(self.label_size, self.hidden_size),
                nn.LeakyReLU()).to(self.device)
        self.lead_generators = [nn.Linear(self.hidden_size*2, self.output_size, bias=False).to(self.device) for _ in range(self.n_lead)]
        
    def forward(self, noise, label):
        noise = self.noise_encoder(noise)
        label = self.label_encoder(label)
        x = torch.cat([noise, label], dim=1)
        generated_lead_list = [torch.tanh(lead_g(x)).unsqueeze(-1) for lead_g in self.lead_generators]
        return torch.cat(generated_lead_list, dim=2)

class ConvCritic(nn.Module):
    def __init__(self, ):
        pass
    
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
            nn.LeakyReLU()
        ).to(self.device)
        self.lead_encoder_list = [
            nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size//2),
                nn.LeakyReLU()
            ).to(self.device) for _ in range(self.lead_n)]
        self.critic = nn.Linear(self.hidden_size, 1).to(self.device)
        
    def forward(self, sequence, label):
        encoded_leads = [lead_encoder(sequence[:,:,i]) for i, lead_encoder in enumerate(self.lead_encoder_list)]
        encoded_label = self.label_encoder(label)
        critic_outputs = [torch.sigmoid(self.critic(torch.cat([lead, encoded_label], dim=1))) for lead in encoded_leads]
        return torch.cat(critic_outputs, dim=1)
        
        

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Get from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module, device):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxD)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module
        self.device = device

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxD
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            # print(x.shape)
            mask = torch.ByteTensor(x.size()).fill_(0).to(self.device)
            # print(mask.shape)
            # if x.is_cuda:
            # mask = mask
            for i, length in enumerate(lengths):
                # length = length.item()
                if (mask[i].size(1) - length) > 0:
                    mask[i].narrow(1, length, mask[i].size(1) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class Generator(nn.Module):
    def __init__(self, labels_dim, hidden_dim, latent_dim,
                 device='cpu',
                 decoder_output_dim=12,
                 num_layers=2,
                 dropout=0.2,
                 batch_first=True,
                 bidirectional=False,
                 batch_size=1,
                 dataset_for_labels=None,
                 unique_diagnosis_labels=None,
                 sample_len_collection=[5000, 7500, 10000, 12500, 15000]
                 ):
        """
        Create a Generator object for time-series generation
        """
        super(Generator, self).__init__()
        self.labels_dim = labels_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device
        self.decoder_output_dim = decoder_output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        if unique_diagnosis_labels is None:
            self.unique_diagnosis_labels = [torch.from_numpy(np.array(_i, dtype=np.float32))
                                            for _i in set([tuple(it[1].numpy().tolist()) for it in dataset_for_labels])]
        self.sample_len_collection = sample_len_collection

        self.lstm_output_mult = 1 + int(self.bidirectional)
        # Define our lstm layer
        self.lstm = nn.LSTM(input_size=self.labels_dim + self.latent_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=self.batch_first,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional,
                            ).to(self.device)
        # Define our decoder layer
        fully_connected = nn.Sequential(
            # nn.BatchNorm1d(self.hidden_dim * self.lstm_output_mult),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * self.lstm_output_mult, self.decoder_output_dim, bias=False)
        )
        self.decoder = fully_connected.to(self.device)
        # self.decoder = nn.Linear(self.hidden_dim * self.lstm_output_mult, self.decoder_output_dim).to(self.device)

    def init_hidden(self):
        return tuple([torch.zeros((self.num_layers, self.batch_size, self.hidden_dim),
                                  dtype=torch.float32).to(self.device)
                      for _ in range(self.num_layers)])

    def forward(self, _input, _hidden):
        """
            if _category == None then _input already combined
        """
        input_combined = _input.to(self.device)

        raw_output, _hidden = self.lstm(input_combined, _hidden)
        if isinstance(raw_output, PackedSequence):
            raw_output, lengths = pad_packed_sequence(raw_output, batch_first=self.batch_first)
        # print(f"[Generator] shape of vector for fc layer {raw_output.shape}")
        decoded_output = self.decoder(raw_output)
        decoded_output = torch.tanh(decoded_output)
        return decoded_output, _hidden

    def get_nllt_list(self, size):
        return [(torch.rand((1, self.latent_dim), dtype=torch.float32).expand(_len, -1).to(self.device),
                 random.choice(self.unique_diagnosis_labels).unsqueeze(0).float().expand(_len, -1).to(self.device),
                 _len)
                for _len in random.choices(self.sample_len_collection, k=size)]

    def generate_seq_batch(self):
        noise_label_len_tuple_list = self.get_nllt_list(self.batch_size)
        sorted_nllt_list = sorted(noise_label_len_tuple_list, key=lambda x: x[2], reverse=True)

        noises = [x[0] for x in sorted_nllt_list]
        labels = [x[1] for x in sorted_nllt_list]
        lenghts = [x[2] for x in sorted_nllt_list]

        noises_padded = pad_sequence(noises, batch_first=self.batch_first)
        labels_padded = pad_sequence(labels, batch_first=self.batch_first)
        #         print(f"{noises_padded.shape}, {labels_padded.shape}")
        labels_noises_padded = torch.cat([labels_padded, noises_padded], 2)

        labels_noises_packed = pack_padded_sequence(labels_noises_padded, lenghts, batch_first=self.batch_first)

        generated_seq, _ = self.forward(labels_noises_packed, self.init_hidden())

        return generated_seq, lenghts, labels_padded


class Discriminator(nn.Module):
    def __init__(self,
                 input_dim,
                 labels_dim,
                 hidden_dim,
                 encoder_dim,
                 decoder_dim,
                 batch_size,
                 device='cpu',
                 num_layers=2,
                 dropout=0.2,
                 batch_first=True,
                 bidirectional=False):
        """
        Create a Discriminator object for time-series clasification -- real or fake
        """
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.labels_dim = labels_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.batch_size = batch_size
        self.device = device
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.lstm_output_mult = 1 + int(self.bidirectional)

        self.encoder = nn.Sequential(
            nn.Linear(self.labels_dim + self.input_dim, self.encoder_dim),
            nn.ReLU()
        ).to(self.device)
        # Define our lstm layer
        self.lstm = nn.LSTM(input_size=self.encoder_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=self.batch_first,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional,
                            ).to(self.device)
        # Define our decoder layer
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * self.lstm_output_mult, self.decoder_dim),
            nn.Sigmoid()
        ).to(self.device)

    def init_hidden(self):
        return tuple([torch.zeros((self.num_layers, self.batch_size, self.hidden_dim),
                                  dtype=torch.float32).to(self.device)
                      for _ in range(self.num_layers)])

    def forward(self, _input, _hidden=None):
        """
            if _category == None then _input already combined
        """
        if _hidden is None:
            _hidden = self.init_hidden()

        encoder_input, lengths = pad_packed_sequence(_input, batch_first=self.batch_first)
        encoded_output = self.encoder(encoder_input)
        packed_encoded_output = pack_padded_sequence(encoded_output, lengths, batch_first=self.batch_first)
        rnn_output, _hidden = self.lstm(packed_encoded_output, _hidden)
        if isinstance(rnn_output, PackedSequence):
            rnn_output, lengths = pad_packed_sequence(rnn_output, batch_first=self.batch_first)
        decoded_output = self.decoder(rnn_output)
        return decoded_output, lengths


class CNNDiscriminator(nn.Module):
    def __init__(self,
                 input_dim,
                 labels_dim,
                 device='cpu'):
        super(CNNDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.labels_dim = labels_dim
        self.device = device

        # depthwise convolution with dilation
        self.conv = MaskConv(nn.Sequential(
            nn.Conv1d(self.input_dim,
                      self.input_dim * 2,
                      kernel_size=4,
                      stride=1,
                      padding=2,
                      groups=self.input_dim,
                      dilation=2),
            # nn.BatchNorm1d(self.input_dim * 2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(self.input_dim * 2,
                      self.input_dim * 4,
                      kernel_size=4,
                      stride=1,
                      padding=2,
                      groups=self.input_dim,
                      dilation=2),
            # nn.BatchNorm1d(self.input_dim * 4),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(self.input_dim * 4,
                      self.input_dim * 4,
                      kernel_size=4,
                      stride=1,
                      padding=2,
                      groups=self.input_dim,
                      dilation=2),
            # nn.BatchNorm1d(self.input_dim * 4),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(self.input_dim * 4,
                      self.input_dim * 6,
                      kernel_size=4,
                      stride=1,
                      padding=2,
                      groups=self.input_dim,
                      dilation=2),
            # nn.BatchNorm1d(self.input_dim * 4),
            nn.MaxPool1d(kernel_size=2)
        ).to(self.device), self.device)
        fc_input_size = self.input_dim * 6
        # print(fc_input_size)
        fully_connected = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(fc_input_size, 1),
        )
        # self.fc = nn.Sequential(SequenceWise(fully_connected)).to(self.device)
        self.fc = nn.Sequential(SequenceWise(fully_connected)).to(self.device)

    def forward(self, x, lengths):
        # print(f"x init shape: {x.shape}")
        if isinstance(x, PackedSequence):
            x, lengths = pad_packed_sequence(x, batch_first=True)
        # x =
        x = x.view(x.size(0), x.size(2), x.size(1))
        # print(f"x shape after reshape: {x.shape}")
        x, lengths = self.conv(x, lengths)
        # print(f"x lengths after conv {lengths}")
        # print(f"x shape after conv: {x.shape}")
        x = x.view(x.size(0), x.size(2), x.size(1))
        # print(f"x shape after conv and reshape: {x.shape}")
        x = self.fc(x)
        x = torch.sigmoid(x)
        # print(x.shape, lengths)
        return x, lengths
