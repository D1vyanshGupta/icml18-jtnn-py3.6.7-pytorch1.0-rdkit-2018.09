import os
import sys
import pickle
import argparse
import warnings
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

import rdkit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# implicitly set PYTHONPATH for the relevant imports
PARENT_DIR_PATH = os.path.abspath('..')
sys.path.insert(0, PARENT_DIR_PATH)

from jtnn import *

# supress warnings
warnings.filterwarnings("ignore", category=UserWarning)

train_path = os.path.join(os.getcwd(), 'zinc_processed')
vocab_path = os.path.join(PARENT_DIR_PATH, 'data', 'vocab.txt')

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--plot_dir', required=True)
parser.add_argument('--model_name', required=True)
parser.add_argument('--log_dir', required=True)
parser.add_argument('--load_epoch', type=int, default=0)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)
parser.add_argument('--share_embedding', action='store_true')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--warmup', type=int, default=20000)

parser.add_argument('--epochs', type=int, default=3)

parser.add_argument('--enable_lr_anneal', action='store_true')
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=20000)

parser.add_argument('--enable_beta_anneal', action='store_true')
parser.add_argument('--step_beta', type=float, default=0.01)
parser.add_argument('--beta_anneal_iter', type=int, default=5000)
parser.add_argument('--max_beta', type=float, default=1.0)

parser.add_argument('--print_iter', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--use_graph_conv', action='store_true')

# DEFAULT_CMD = ["--train", "zinc_processed/", "--vocab", "../data/vocab.txt"]

args = parser.parse_args()
print(args)

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = ClusterVocab(vocab)

# model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG, args.num_layers, args.use_graph_conv, args.share_embedding)
model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG, args.num_layers, args.use_graph_conv, args.share_embedding).cuda()
print(model)

# for all multi-dimensional parameters, initialize them using xavier initialization
# for one-dimensional parameters, initialize them to 0.
for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch > 0:
    model.load_state_dict(torch.load(args.save_dir + "/model.epoch-" + str(args.load_epoch)))

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

# use adam optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# exponentially decay the learning rate
if args.enable_lr_anneal:
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    scheduler.step()

# param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
# grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = args.load_epoch
beta = args.beta
meters = np.zeros(6)

# list to store loss and accuracies
loss_lst = []
kl_div_lst = []
wacc_lst = []
tacc_lst = []
aacc_lst = []
sacc_lst = []

total_training_time = 0

for epoch in range(args.epochs):
    epoch_time = 0
    loader = MolTreeFolder(args.train, vocab, args.use_graph_conv, args.batch_size, num_workers=5)
    for idx, batch in enumerate(loader):
        total_step += 1
        try:
            start = timer()

            # reset the gradient buffer to 0.
            model.zero_grad()
            # implement forward pass
            loss, kl_div, wacc, tacc, aacc, sacc = model(batch, beta)

            # append items to list
            loss_lst.append(loss.item())
            kl_div_lst.append(kl_div)
            wacc_lst.append(wacc)
            tacc_lst.append(tacc)
            aacc_lst.append(aacc)
            sacc_lst.append(sacc)

            # implement backpropagation
            loss.backward()

            # implement gradient clipping
            # nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            # update model parameters
            optimizer.step()

            end = timer()

            time_for_iteration = (end - start)

            epoch_time += time_for_iteration

        except Exception as e:
            print(e)
            continue

        meters = meters + np.array([loss.item(), kl_div, wacc * 100, tacc * 100, aacc * 100, sacc * 100])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            print("Epoch:{}, Iteration: {}, Step: {}, Beta: {:.3f}, Loss:{:.2f}, KL: {:.2f}, Word: {:.2f}, Topo: {:.2f}, Assm: {:.2f}, Stereo: {:.2f}, Time: {:.2f} min".format(
                    epoch + 1, idx + 1, total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], time_for_iteration / 60))
            sys.stdout.flush()
            meters *= 0

        # implement learning-rate annealing
        if args.enable_lr_anneal and total_step % args.anneal_iter == 0:
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

        # implement beta-annealing
        if args.enable_beta_anneal and total_step % args.beta_anneal_iter == 0 and total_step >= args.warmup:
            beta = min(args.max_beta, beta + args.step_beta)

    # save the model after this epoch
    model_name = (args.model_name).lower().replace(' ', '_')
    torch.save(model.state_dict(), args.save_dir + "/{}.epoch-".format(model_name) + str(epoch + args.load_epoch + 1))

    print('Epoch: {} completed. Time taken: {:.2f} min.'.format(epoch + args.load_epoch + 1, epoch_time / 60))
    total_training_time += epoch_time

print('Total Training Time: {:.2f} min.'.format(total_training_time / 60))
# create model info
model_info_line_1 = "Hidden Size: {}, Latent Size: {}, DepthG: {}, DepthT: {}".format(args.hidden_size, args.latent_size, args.depthG, args.depthT)
model_info_line_2 = "Batch Size: {}, Initial LR: {}, Initial Beta: {}, Max Beta: {}".format(args.batch_size, args.lr, args.beta, args.max_beta)

if args.enable_lr_anneal:
    model_info_line_3 = "LR Anneal Rate: {}, LR Anneal Iter: {}".format(args.anneal_rate, args.anneal_iter)
else:
    model_info_line_3 = "LR Annealing Disabled"

if args.enable_beta_anneal:
    model_info_line_4 = "Beta Annealing Begins at: {}, Step Beta: {}, Beta Anneal Iter: {}".format(args.warmup,   args.step_beta, args.beta_anneal_iter)
else:
    model_info_line_4 = "Beta Annealing Disabled"

model_info = "\n" + model_info_line_1 + "\n" + model_info_line_2 + "\n" + model_info_line_3 + "\n" + model_info_line_4

# figure for loss and kl-divergence
# plot graphs
fig = plt.figure(0, figsize=(15, 10))
plt.plot(np.arange(total_step), loss_lst, label='Overall Loss')
plt.plot(np.arange(total_step), kl_div_lst, label='KL-Divergence')

plt.title(args.model_name + " " + "Losses" + model_info, fontsize=10, pad=15.0)

plt.xlabel('Iterations', fontsize=20, labelpad=10)
plt.ylabel('Loss', fontsize=20, labelpad=10)

plt.tick_params(labelsize=20)
plt.legend()

plot_name = (args.model_name).lower().replace(' ', '_') + "_loss"

fig_path = os.path.join(PARENT_DIR_PATH, 'plots', plot_name + '.png')
plt.savefig(fig_path, dpi=200)
plt.close(fig)

fig = plt.figure(0, figsize=(15, 10))
plt.plot(np.arange(total_step), tacc_lst, label='Topological Accuracy')
plt.plot(np.arange(total_step), wacc_lst, label='Label Accuracy')
plt.plot(np.arange(total_step), sacc_lst, label='Stereo Accuracy')
plt.plot(np.arange(total_step), aacc_lst, label='Assembly Accuracy')

plt.title(args.model_name + " " + "Accuracies" + model_info, fontsize=10, pad=15.0)

plt.xlabel('Iterations', fontsize=20, labelpad=10)
plt.ylabel('Accuracy', fontsize=20, labelpad=10)

plt.tick_params(labelsize=20)
plt.legend()

plot_name = (args.model_name).lower().replace(' ', '_') + "_acc"

fig_path = os.path.join(args.plot_dir, plot_name + '.png')
plt.savefig(fig_path, dpi=200)
plt.close(fig)

# save various lists
model_name = (args.model_name).lower().replace(' ', '_')

with open(args.log_dir + '/{}_loss_lst.pkl'.format(model_name), 'wb') as f:
    pickle.dump(loss_lst, f, pickle.HIGHEST_PROTOCOL)

with open(args.log_dir + '/{}_kl_div_lst.pkl'.format(model_name), 'wb') as f:
    pickle.dump(kl_div_lst, f, pickle.HIGHEST_PROTOCOL)

with open(args.log_dir + '/{}_wacc_lst.pkl'.format(model_name), 'wb') as f:
    pickle.dump(wacc_lst, f, pickle.HIGHEST_PROTOCOL)

with open(args.log_dir + '/{}_tacc_lst.pkl'.format(model_name), 'wb') as f:
    pickle.dump(tacc_lst, f, pickle.HIGHEST_PROTOCOL)

with open(args.log_dir + '/{}_aacc_lst.pkl'.format(model_name), 'wb') as f:
    pickle.dump(aacc_lst, f, pickle.HIGHEST_PROTOCOL)

with open(args.log_dir + '/{}_sacc_lst.pkl'.format(model_name), 'wb') as f:
    pickle.dump(sacc_lst, f, pickle.HIGHEST_PROTOCOL)
