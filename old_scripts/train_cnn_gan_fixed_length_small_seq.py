#  key concept - sygnal normalization
#         curr_lead_data = data_dict[patient][lead_n, :]/np.abs(data_dict[patient][lead_n, :]).max()
import os
import pickle
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.data import ECGDataset, pad_batch_sequence, save_ecg_example
from utils.models import Generator, Discriminator, CNNDiscriminator


def get_argument_parser():
    # TODO description to all params
    # parameters for training
    parser = ArgumentParser()
    parser.add_argument("--out_dir",
                        help="Path to dir where models checkpoints, generated examples "
                             "and tensorboard logs will be stored",
                        type=str,
                        default="./out")
    parser.add_argument("--model_name", default="cnn_gan_fixed_length_small_seq")
    # dataset params
    parser.add_argument("--real_dataset",
                        help="Path to .pickle file with ecg data from prepare_data script",
                        type=str)
    parser.add_argument("--real_labels",
                        help="Path to .csv file with diagnosis labels from prepare_data script",
                        type=str)
    parser.add_argument("--labels_dim", default=7, type=int)
    parser.add_argument("--lead_n", default=12, type=int)
    # general params
    parser.add_argument("--device",
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    # generator params
    parser.add_argument("--gen_h_dim", default=128, type=int)
    parser.add_argument("--gen_l_dim", default=100, type=int)
    # discriminator params
    parser.add_argument("--dis_h_dim", default=128, type=int)
    parser.add_argument("--dis_encoder_h_dim", default=128, type=int)
    return parser


if __name__ == "__main__":
    args = get_argument_parser().parse_args()

    os.makedirs(os.path.join(args.out_dir, 'pictures'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'models'), exist_ok=True)
    tb_path = os.path.join(args.out_dir, 'tensorboard')
    os.makedirs(tb_path, exist_ok=True)

    tb_writer = SummaryWriter(os.path.join(tb_path, args.model_name))

    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(123)
    torch.manual_seed(123)
    # load our real examples
    real_data_dict = pickle.load(open(args.real_dataset, 'rb'))
    real_dataset = ECGDataset(real_data_dict, args.real_labels)
    # loss and data loader setup
    criterion = nn.BCELoss()
    real_data_loader = DataLoader(real_dataset, batch_size=args.batch_size, drop_last=True,
                                  collate_fn=pad_batch_sequence, shuffle=True)
    # init GAN models
    G = Generator(labels_dim=args.labels_dim,
                  hidden_dim=args.gen_h_dim,
                  latent_dim=args.gen_l_dim,
                  batch_size=args.batch_size,
                  device=args.device,
                  decoder_output_dim=args.lead_n,
                  dataset_for_labels=real_dataset,
                  sample_len_collection=[500])
    D = CNNDiscriminator(input_dim=args.lead_n + args.labels_dim,
                         labels_dim=args.labels_dim,
                         device=args.device)

    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

    # train loop
    for epoch in tqdm.tqdm(range(args.n_epochs), position=0):
        # sequences in true_data_loader already padded thanks to pad_batch_sequence function
        for real_seqs, real_lengths, real_labels in tqdm.tqdm(real_data_loader, position=1):
            torch.cuda.empty_cache()
            """------------------------------ Discriminator step --------------------------------------"""
            # Generate fake sample,
            fake_seq_padded_batch, fake_lengths, fake_labels_padded_batch = G.generate_seq_batch()
            # Let's prepare our fake and real samples
            # concat labels + sequence alongside
            fake_label_seq_padded_batch = torch.cat([fake_labels_padded_batch, fake_seq_padded_batch], 2)
            # fake_packed_batch = pack_padded_sequence(fake_label_seq_padded_batch, fake_lengths, batch_first=True)
            # and the same for real ones
            real_label_seq_padded_batch = torch.cat([real_labels, real_seqs], 2)
            # real_packed_batch = pack_padded_sequence(real_label_seq_padded_batch, real_lengths, batch_first=True)
            # After that we can make predictions for our fake examples
            d_fake_predictions, _ = D(fake_label_seq_padded_batch, fake_lengths)
            d_fake_target = torch.zeros_like(d_fake_predictions)
            # ... and real ones
            d_real_predictions, real_pred_lengths = D(real_label_seq_padded_batch.to(args.device), real_lengths)
            d_real_target = torch.ones_like(d_real_predictions)
            # Now we can calculate loss for discriminator
            # TODO Need to make sure that sequence with vary length is ok for calc loss
            d_fake_loss = criterion(d_fake_predictions, d_fake_target)
            d_real_loss = criterion(d_real_predictions, d_real_target)
            d_loss = d_real_loss + d_fake_loss
            tb_writer.add_scalar("D_loss", d_loss.item(), global_step=epoch)
            # And make back-propagation according to calculated loss
            d_loss.backward()
            D_optimizer.step()
            # Housekeeping - reset gradient
            D_optimizer.zero_grad()
            G.zero_grad()
            """ ---------------------------- Generator step ---------------------------------------------"""
            # Generate fake sample
            fake_seq_padded_batch, fake_lengths, fake_labels_padded_batch = G.generate_seq_batch()
            # concat labels + sequence alongside
            fake_label_seq_padded_batch = torch.cat([fake_labels_padded_batch, fake_seq_padded_batch], 2)
            # fake_packed_batch = pack_padded_sequence(fake_label_seq_padded_batch, fake_lengths, batch_first=True)
            # After that we can make predictions for our fake examples
            d_fake_predictions, _ = D(fake_label_seq_padded_batch, fake_lengths)
            g_target = torch.ones_like(d_fake_predictions)
            # Now we can calculate loss for generator
            g_loss = criterion(d_fake_predictions, g_target)
            tb_writer.add_scalar("G_loss", g_loss.item(), global_step=epoch)
            # And make back-propagation according to calculated loss
            g_loss.backward()
            G_optimizer.step()
            # Housekeeping - reset gradient
            G_optimizer.zero_grad()
            D.zero_grad()
        # plot example and save checkpoint each odd epoch
        if epoch % 2 == 1:
            print(f'Epoch-{epoch}; D_loss: {d_loss.data.cpu().numpy()}; G_loss: {g_loss.data.cpu().numpy()}')
            torch.save({
                'epoch': epoch,
                "d_model": D,
                "d_loss": d_loss,
                "d_optimizer": D_optimizer,
                "g_model": G,
                "g_loss": g_loss,
                "g_optimizer": G_optimizer,
            }, os.path.join(args.out_dir, f"models/{args.model_name}_epoch_{epoch}_checkpoint.pkl"))
            with torch.no_grad():
                _seqs, _lengths, _labels = G.generate_seq_batch()
                _seq = _seqs[0].cpu().numpy()  # batch_first :^)
                _label = _labels[0].cpu().numpy()
                fig = save_ecg_example(_seq, f"{args.model_name}_epoch_{epoch}_example")
                tb_writer.add_figure("generated_example", fig, global_step=epoch)

                # TODO use visualize func here
