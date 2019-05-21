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
from utils.models import InverseCNNGenerator, ConvCritic


def get_argument_parser():
    # TODO description to all params
    # parameters for training
    parser = ArgumentParser()
    parser.add_argument("--out_dir",
                        help="Path to dir where models checkpoints, generated examples "
                             "and tensorboard logs will be stored",
                        type=str,
                        default="./out")
    parser.add_argument("--model_name", default="dense_gan_cold_start")
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
    parser.add_argument("--gen_h_dim", default=256, type=int)
    parser.add_argument("--gen_l_dim", default=100, type=int)
    # discriminator params
    parser.add_argument("--dis_h_dim", default=256, type=int)
#     parser.add_argument("--dis_encoder_h_dim", default=128, type=int)
    parser.add_argument("--continue_from", default=None, type=str)
    parser.add_argument("--seq_len", default=500, type=int)
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
    # init GAN models
    if args.continue_from:
        G = torch.load(open(args.continue_from, "rb"), map_location=args.device)['g_model']
        G.device = args.device
#         G.sample_len_collection = [500]
        D = torch.load(open(args.continue_from, "rb"), map_location=args.device)['d_model']
        D.device = args.device
    else:
        G = InverseCNNGenerator(input_size=args.gen_l_dim, 
                        label_size=args.labels_dim,
                        lead_n=args.lead_n, 
                        hidden_size=args.gen_h_dim, 
                        stride=1,
                        padding=2,
                        dilation=2,
                        kernel_size=7,
                        out_padding=1,
                        device=args.device
                       )
        noise = torch.rand(8, args.gen_l_dim, device=args.device)
        labels = torch.randint(0,1,(8, args.labels_dim), device=args.device).float()
        result = G(noise=noise, label=labels)
        true_seq_len = result.size(1)
        D = ConvCritic(input_size=true_seq_len,
                       leads_n=args.lead_n, 
                       label_size=args.labels_dim, 
                       hidden_size=args.dis_h_dim, 
                       device=args.device)

    real_data_dict = pickle.load(open(args.real_dataset, 'rb'))
    real_dataset = ECGDataset(real_data_dict, args.real_labels, seq_len=true_seq_len)
    # loss and data loader setup
    criterion = nn.BCELoss()
    real_data_loader = DataLoader(real_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)

    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)
    
    # get fixed bath for hot-start
    fixed_seqs, fixed_labels = [it for it in real_data_loader][0]
    fixed_seqs = fixed_seqs.to(args.device)
    fixed_labels = fixed_labels.to(args.device)
    # then we need to try learn shape of our signal
    loss = nn.MSELoss()
    for prem_epoch in tqdm.tqdm(range(10000)):
        noise = torch.rand(args.batch_size, args.gen_l_dim, device=args.device)
        fake_seq = G(noise, fixed_labels)
        output = loss(fake_seq, fixed_seqs)
        tb_writer.add_scalar("D_hot_start_mse_loss", output.item(), global_step=prem_epoch)
        output.backward()
        G_optimizer.step()
        G_optimizer.zero_grad()
    torch.save({
#                 'epoch': epoch,
                "d_model": D,
#                 "d_loss": d_loss,
                "d_optimizer": D_optimizer,
                "g_model": G,
#                 "g_loss": g_loss,
                "g_optimizer": G_optimizer,
            }, os.path.join(args.out_dir, f"models/{args.model_name}_after_hot_start_checkpoint.pkl"))
    with torch.no_grad():
        noise = torch.rand(args.batch_size, args.gen_l_dim, device=args.device)
        fake_seq = G(noise, fixed_labels)
        _seq = fake_seq[0].cpu().numpy()  # batch_first :^)
        fig = save_ecg_example(_seq, f"{args.model_name}_epoch_0_example")
        tb_writer.add_figure("generated_example", fig, global_step=0)

    # train loop
    for epoch in tqdm.tqdm(range(args.n_epochs), position=0):
        # sequences in true_data_loader already padded thanks to pad_batch_sequence function
        for real_seqs, real_labels in tqdm.tqdm(real_data_loader, position=1):
            torch.cuda.empty_cache()
            """------------------------------ Discriminator step --------------------------------------"""
            # Generate fake sample,
            real_seqs = real_seqs.to(args.device)
            real_labels =  real_labels.to(args.device)
            noise = torch.rand(args.batch_size, args.gen_l_dim, device=args.device)
            fake_seq = G(noise, real_labels)
            # After that we can make predictions for our fake examples
            d_fake_predictions = D(fake_seq, real_labels)
            d_fake_target = torch.zeros_like(d_fake_predictions)
            # ... and real ones
            d_real_predictions = D(real_seqs, real_labels)
            d_real_target = torch.ones_like(d_real_predictions)
            # Now we can calculate loss for discriminator
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
            noise = torch.rand(args.batch_size, args.gen_l_dim, device=args.device)
            fake_seq = G(noise, real_labels)
            # After that we can make predictions for our fake examples
            d_fake_predictions = D(fake_seq, real_labels)
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
        if epoch % 250 == 249:
            print(f"Epoch-{epoch}; D_loss: {d_loss.data.cpu().numpy()}; G_loss: {g_loss.data.cpu().numpy()}")
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
                noise = torch.rand(args.batch_size, args.gen_l_dim, device=args.device)
                fake_seq = G(noise, real_labels)
                _seq = fake_seq[0].cpu().numpy()  # batch_first :^)
                fig = save_ecg_example(_seq, f"{args.model_name}_epoch_{epoch}_example")
                tb_writer.add_figure("generated_example", fig, global_step=epoch)
