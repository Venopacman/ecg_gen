import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.nn.functional as F

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
        self.decoder = nn.Linear(self.hidden_dim * self.lstm_output_mult, self.decoder_output_dim).to(self.device)

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
        decoded_output = self.decoder(raw_output)
        decoded_output = F.sigmoid(decoded_output)
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

        #         if _category:
        #             input_combined = torch.cat((_category, _input), 2)
        #         else:
        #             input_combined = _input

        encoder_input, lengths = pad_packed_sequence(_input, batch_first=self.batch_first)

        encoded_output = self.encoder(encoder_input)

        packed_encoded_output = pack_padded_sequence(encoded_output, lengths, batch_first=self.batch_first)

        rnn_output, _hidden = self.lstm(packed_encoded_output, _hidden)
        if isinstance(rnn_output, PackedSequence):
            rnn_output, lengths = pad_packed_sequence(rnn_output, batch_first=self.batch_first)

        decoded_output = self.decoder(rnn_output)
        return decoded_output, lengths
