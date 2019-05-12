from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch

from utils.data import save_ecg_example
from utils.models import Generator, Discriminator


def get_argument_parser():
    parser = ArgumentParser()
    parser.add_argument("--device",
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return parser


if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    checkpoint_dict = torch.load("./out/models/epoch_85_checkpoint.pkl", map_location=args.device)
    G: Generator = checkpoint_dict['g_model']
    G.device = args.device
    with torch.no_grad():
        seqs, lengths, labels = G.generate_seq_batch()
        _seqs, _lengths, _labels = G.generate_seq_batch()
        _seq = _seqs[0].cpu().numpy()  # batch_first :^)
        _label = _labels[0].cpu().numpy()
        fig = save_ecg_example(_seq, f"test_85")
        plt.show(fig)
