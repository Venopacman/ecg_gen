import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, patients_dict: dict, labels_filepath: str, transform_strategy='cut'):
        self.transform_strategy = transform_strategy
        self.labels_filepath = labels_filepath
        self.dataset = self.get_pairset(patients_dict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform_strategy:
            sample = self.transform(sample)
        return sample

    def get_pairset(self, patients_dict: dict):
        labels_df = self.parse_labels_file()
        return [(torch.Tensor(patients_dict[f"{patient}"].T),
                 torch.Tensor(labels_df.loc[patient].values.astype(bool).astype(int).astype(float)))
                for patient in labels_df.index]

    def transform(self, sample):
        # TODO 1st make cut transform_strategy
        return sample

    def parse_labels_file(self):
        return pd.read_csv(self.labels_filepath, header=0).set_index('patient')


def pad_batch_sequence(batch):
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = pad_sequence(sequences, batch_first=True)
    lengths = [len(x) for x in sequences]
    # Don't forget to grab the labels of the *sorted* batch
    labels = [x[1].unsqueeze(0).expand(len(x[0]), -1) for x in sorted_batch]
    labels_padded = pad_sequence(labels, batch_first=True)
    return sequences_padded, lengths, labels_padded


def save_ecg_example(gen_data: np.array, image_name, image_title='12-lead ECG'):
    """
    Save 12-lead ecg signal in fancy .png
    :param gen_data:
    :param image_name:
    :param image_title:
    :return:
    """
    fig = plt.figure(figsize=(12, 14))
    for _lead_n in range(gen_data.shape[1]):
        curr_lead_data = gen_data[:, _lead_n]
        plt.subplot(4, 3, _lead_n + 1)
        plt.plot(curr_lead_data, label=f'lead_{_lead_n + 1}')
        plt.title(f'lead_{_lead_n + 1}')
    fig.suptitle(image_title)
    plt.savefig(f'out/{image_name}.png', bbox_inches='tight')
    plt.close(fig)
    return fig
