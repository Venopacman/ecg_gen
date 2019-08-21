import os
import pickle
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.data import pad_batch_sequence, save_ecg_example, UnlabeledECGDataset, ECGRecord
from utils.models import UnlabeledLSTMGenerator, UnlabeledConvCritic
from torch_two_sample.statistics_diff import MMDStatistic

def get_argument_parser():
    # parameters for training
    parser = ArgumentParser()
    parser.add_argument("--out_dir",
                        help="Path to dir where models checkpoints, generated examples "
                             "and tensorboard logs will be stored",
                        type=str,
                        default="./out")
    parser.add_argument("--model_name", default="lstm_vs_cnn_gan_train_unlabeled_lr_5e-3")
    # dataset params
    parser.add_argument("--dataset",
                        help="Path to dir with subdirs with csv files with unlabeled ecg",
                        default="../data/",
                        type=str)
    parser.add_argument("--lead_n", default=12, type=int)
    # general params
    parser.add_argument("--device",
                        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--n_epochs", default=2000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    # generator params
    parser.add_argument("--gen_h_dim", default=256, type=int)
    parser.add_argument("--gen_l_dim", default=5000, type=int)
    # discriminator params
    parser.add_argument("--dis_h_dim", default=256, type=int)
    # continue params
    parser.add_argument("--continue_from", default=None, type=str)
    parser.add_argument("--seq_len", default=5000, type=int)
    return parser


def get_all_csv_files_within_dir(dir_path: str):
    return [os.path.join(curr_dir, file) 
                for curr_dir, dir_list, file_list in os.walk(dir_path) 
                    for file in file_list 
                    if file.endswith(".csv") and curr_dir != dir_path ]


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
    
    real_ecg_records_list = [ ECGRecord(it) 
                             for it in tqdm.tqdm([i for i in get_all_csv_files_within_dir(args.dataset)[:1000]]) ]
    
    real_ecg_records_list = [it for it in real_ecg_records_list if it.ecg_signal is not None]
    real_dataset = UnlabeledECGDataset(real_ecg_records_list, seq_len=args.seq_len)
    # loss and data loader setup
    criterion = nn.BCELoss()
    mmd = MMDStatistic(args.batch_size, args.batch_size)
    real_data_loader = DataLoader(real_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
    
    # init GAN models
    if args.continue_from:
        G = torch.load(open(args.continue_from, "rb"), map_location=args.device)['g_model']
        G.device = args.device
        D = torch.load(open(args.continue_from, "rb"), map_location=args.device)['d_model']
        D.device = args.device
    else:
        G = UnlabeledLSTMGenerator(noise_size=args.seq_len,
                                   hidden_size=args.gen_h_dim,
                                   n_lead=args.lead_n,
                                   device=args.device)
        D = UnlabeledConvCritic(input_size=args.seq_len, 
                                hidden_size=args.dis_h_dim,
                                lead_n=args.lead_n, 
                                device=args.device)

    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)
    
    best_stat = None
    is_graphs_not_saved = True
    # train loop
    for epoch in tqdm.tqdm(range(args.n_epochs), position=0):
        # sequences in true_data_loader already padded thanks to pad_batch_sequence function
        stat_list = []
        for real_seqs in tqdm.tqdm(real_data_loader, position=1):
            torch.cuda.empty_cache()
            """------------------------------ Discriminator step --------------------------------------"""
            # Generate fake sample,
            real_seqs = real_seqs.to(args.device)
            noise = torch.rand(args.batch_size, args.seq_len, 1, device=args.device)
            fake_seq = G(noise)
            
            # After that we can make predictions for our fake examples
            d_fake_predictions = D(fake_seq)
            d_fake_target = torch.zeros_like(d_fake_predictions)
            # ... and real ones
            d_real_predictions = D(real_seqs)
            d_real_target = torch.ones_like(d_real_predictions)
            
            # Now we can calculate loss for discriminator
            d_fake_loss = criterion(d_fake_predictions, d_fake_target)
            d_real_loss = criterion(d_real_predictions, d_real_target)
            d_loss = d_real_loss + d_fake_loss
            
            statistic = mmd(fake_seq.view(args.batch_size, -1), real_seqs.view(args.batch_size, -1), [1.])
            stat_list.append(statistic.item())
            tb_writer.add_scalar("D_loss", d_loss.item(), global_step=epoch)
            # And make back-propagation according to calculated loss
            d_loss.backward()
            D_optimizer.step()
            # Housekeeping - reset gradient
            D_optimizer.zero_grad()
            G.zero_grad()
            
            """ ---------------------------- Generator step ---------------------------------------------"""
            # Generate fake sample
            noise = torch.rand(args.batch_size, args.seq_len, 1, device=args.device)
            fake_seq = G(noise)
            
            # After that we can make predictions for our fake examples
            d_fake_predictions = D(fake_seq)
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
            
            if is_graphs_not_saved:
                tb_writer.add_graph(D, fake_seq)
                tb_writer.add_graph(G, noise)
                is_graphs_not_saved = False
        # plot example and save checkpoint each odd epoch
        if best_stat is None:
#             print()
            best_stat = np.mean(stat_list)
            torch.save({
                'epoch': epoch,
                'stat_value': best_stat,
                "d_model": D,
                "d_loss": d_loss,
                "d_optimizer": D_optimizer,
                "g_model": G,
                "g_loss": g_loss,
                "g_optimizer": G_optimizer,
            }, os.path.join(args.out_dir, f"models/{args.model_name}_best_for_mmd_checkpoint.pkl"))
        elif np.mean(stat_list) < best_stat:
            best_stat = np.mean(stat_list)
            torch.save({
                'epoch': epoch,
                'stat_value': best_stat,
                "d_model": D,
                "d_loss": d_loss,
                "d_optimizer": D_optimizer,
                "g_model": G,
                "g_loss": g_loss,
                "g_optimizer": G_optimizer,
            }, os.path.join(args.out_dir, f"models/{args.model_name}_best_for_mmd_checkpoint.pkl"))
        tb_writer.add_scalar("MMD", np.mean(stat_list), global_step=epoch)
        if epoch % 25 == 0 or epoch == args.n_epochs:
            print(f'Epoch-{epoch}; D_loss: {d_loss.data.cpu().numpy()}; G_loss: {g_loss.data.cpu().numpy()}')
            torch.save({
                'epoch': epoch,
                'stat_value': np.mean(stat_list),
                "d_model": D,
                "d_loss": d_loss,
                "d_optimizer": D_optimizer,
                "g_model": G,
                "g_loss": g_loss,
                "g_optimizer": G_optimizer,
            }, os.path.join(args.out_dir, f"models/{args.model_name}_epoch_{epoch}_checkpoint.pkl"))
            with torch.no_grad():
                noise = torch.rand(args.batch_size, args.seq_len, 1, device=args.device)
                fake_seq = G(noise)
                _seq = fake_seq[0].cpu().numpy()  # batch_first :^)
#                 _label = _labels[0].cpu().numpy()
                fig = save_ecg_example(_seq, f"pictures/{args.model_name}_epoch_{epoch}_example")
                tb_writer.add_figure("generated_example", fig, global_step=epoch)
