import os, sys
from argparse import ArgumentParser

# implicitly set PYTHONPATH for the relevant imports
PARENT_DIR_PATH = os.path.abspath('..')
sys.path.insert(0, PARENT_DIR_PATH)

# supress PyTorch warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt

import rdkit

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader


from jtnn import *

def fun(lst):
    max_len = max([len(x) for x in lst])
    for x in lst:
        if len(x) < max_len:
            num = max_len - len(x)
            x.extend([-1] * num)
    new_lst = []
    for x in lst:
        n_arr = np.array(x)
        # new_lst.append(torch.from_numpy(n_arr).cuda())
        new_lst.append(torch.from_numpy(n_arr))

    new_lst = torch.stack(new_lst, dim=0)

    return new_lst


# supress rdkit warnings
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

# instantiate the define the arguments of the ArgumentParser
parser = ArgumentParser(description="Train JTVAE without KL Regularization.")
parser.add_argument('-t', '--train', action='store', help='Path to the training data file', dest="train_path", required=True)
parser.add_argument('-v', '--vocab', action='store', help='Path to the cluster vocabulary data file', dest="vocab_path", required=True)
parser.add_argument('-s', '--save_dir', action='store', help='Path where the trained model has to be saved.', dest="save_path", required=True)
parser.add_argument("-b", "--batch", action='store', help='The batch size to be used for mini-batch GD.', dest="batch_size", default=40)
parser.add_argument("-w", "--hidden", action='store', help='The size of the hidden message vectors to be used in the model.', dest="hidden_size", default=200)
parser.add_argument("-l", "--latent", action='store', help='The dimension of the latent space to be used.', dest="latent_size", default=56)
parser.add_argument("-d", "--depth", action='store', help='The depth (number of timesteps) for which to run the message passing.', dest="depth", default=3)
parser.add_argument("-n", "--num_layers", action='store', help='The number of layers in the graph convolutional network.', dest="num_layers", default=2)
parser.add_argument("-gc", "--graph_conv", action='store_true', help='Whether to use graph convolutional network or the original JTVAE.', dest="use_graph_conv")
parser.add_argument("-p", "--plot_name", action='store', help='Name of the matplotlib file.', dest="plot_name", required=True)
parser.add_argument("-e", "--epochs", action='store', help='Number of epochs for which to run the model.', dest="epochs", default = 3)
parser.add_argument("-pt", "--plot_title", action='store', help='Title of the plot.', dest="plot_title")

# parse the command line arguments
args = parser.parse_args()

# read the cluster vocabulary from the vocab file
# VOCAB_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data', 'vocab.txt')
TRAIN_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data', 'train_5.txt')
vocab = [x.strip("\r\n ") for x in open(args.vocab_path)]
# vocab = [x.strip("\r\n ") for x in open(VOCAB_PATH)]
vocab = ClusterVocab(vocab)
#
# batch_size = int(args.batch_size)
batch_size = int(2)
hidden_size = int(args.hidden_size)
latent_size = int(args.latent_size)
depth = int(args.depth)
num_layers = int(args.num_layers)
use_graph_conv = args.use_graph_conv

# batch_size = int(2)
# hidden_size = int(450)
# latent_size = int(56)
# depth = int(3)
# num_layers = int(2)
# use_graph_conv = False

device = torch.device("cuda: 0")

model = JTNNVAE(vocab, hidden_size, latent_size, depth, num_layers, use_graph_conv=use_graph_conv)
model = nn.DataParallel(model)

model.to(device)

# initilize all 1-dimensional parameters to 0
# initialize all multi-dimensional parameters by xavier initialization
for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

# use Adam optimizer (apparently the best)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# exponential decay learning rate
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler.step()

# dataset = MoleculeDataset(args.train_path)
dataset = MoleculeDataset(TRAIN_PATH)

# MAX_EPOCH = 3
# NUM_EPOCHS = int(args.epochs)
NUM_EPOCHS = 3
# PRINT_ITER = 20

loss_lst = []
label_pred_loss_lst = []
topo_loss_lst = []
assm_loss_lst = []
stereo_loss_lst = []

# class RandomDataset(Dataset):
#
#     def __init__(self, size, length):
#         self.len = length
#         self.data = torch.randn(length, size)
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#     def __len__(self):
#         return self.len
#
# rand_loader = DataLoader(dataset=RandomDataset(4, 2),
#                          batch_size=2, shuffle=True)
#
# for data in rand_loader:
#     print('rand_loader')
#     print(data.shape)

for epoch in range(NUM_EPOCHS):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=lambda x: x,
                            drop_last=True)

    loss_val, label_pred_loss, topo_loss, assm_loss, stereo_loss = 0, 0, 0, 0, 0

    for it, batch in enumerate(dataloader):

        # for junc_tree in batch:
        #     for node in junc_tree.nodes:
        #         if node.label not in node.candidates:
        #             node.candidates.append(node.label)
        #             node.candidate_mols.append(node.label_mol)

        # flush the gradient buffer
        model.zero_grad()

        # obtain all the losses
        new_batch = fun(batch)
        # new_batch = new_batch.unsqueeze(0)
        print('fun')
        print(new_batch.shape)
        # print(new_batch)
        input = new_batch.to(device)
        # loss, kl_div, label_pred_loss_, topo_loss_, assm_loss_, stereo_loss_ = model(new_batch)
        loss = model(input)

        # print("Epoch: {}, Iteration: {}, loss: {}, label_pred_loss: {}, topo_loss: {}, assm_loss: {}, stereo_loss: {}".format(
        #     epoch + 1, it + 1, loss.item(), label_pred_loss_, topo_loss_, assm_loss_, stereo_loss_.item()
        # ))

        print(loss)
        print(loss.shape)
        print("Epoch: {}, Iteration: {}, loss: {}".format(epoch + 1, it + 1, loss.data))

        # loss_val += loss.item()
        # label_pred_loss += label_pred_loss_
        # topo_loss += topo_loss_
        # assm_loss += assm_loss_
        # stereo_loss += stereo_loss_.item()

        # backpropagation
        loss.backward()

        # update parameters
        optimizer.step()

        # if (it + 1) % PRINT_ITER == 0:
        #     word_acc = word_acc / PRINT_ITER * 100
        #     topo_acc = topo_acc / PRINT_ITER * 100
        #     assm_acc = assm_acc / PRINT_ITER * 100
        #     steo_acc = steo_acc / PRINT_ITER * 100
        #
        #     print("KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f" % (kl_div, word_acc, topo_acc, assm_acc, steo_acc))
        #
        #     word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0
        #     sys.stdout.flush()

    # loss_val /= (it + 1)
    # label_pred_loss /= (it + 1)
    # topo_loss /= (it + 1)
    # assm_loss /= (it + 1)
    # stereo_loss /= (it + 1)

    # print()
    # print('Epoch: {}, loss: {}, label_pred_loss: {}, topo_loss: {}, assm_loss: {}, stereo_loss: {}'.format(
    #     epoch + 1, loss_val, label_pred_loss, topo_loss, assm_loss, stereo_loss
    # ))

    # loss_lst.append(loss_val)
    # label_pred_loss_lst.append(label_pred_loss)
    # topo_loss_lst.append(topo_loss)
    # assm_loss_lst.append(assm_loss)
    # stereo_loss_lst.append(stereo_loss)

    scheduler.step()

    # print("learning rate: %.6f" % scheduler.get_lr()[0])

# torch.save(model.state_dict(), args.save_path + "/model.pre_train")

# plot graphs
# figure = plt.figure(0, figsize=(15, 10))
# plt.plot(np.arange(NUM_EPOCHS), loss_lst, label='overall_loss')
# plt.plot(np.arange(NUM_EPOCHS), label_pred_loss_lst, label='label_pred_loss')
# plt.plot(np.arange(NUM_EPOCHS), topo_loss_lst, label='topo_loss')
# plt.plot(np.arange(NUM_EPOCHS), assm_loss_lst, label='assm_loss')
# plt.plot(np.arange(NUM_EPOCHS), stereo_loss_lst, label='stereo_loss')
#
# plt.xlabel('epochs', fontsize=20, labelpad=10)
# plt.ylabel('loss', fontsize=20, labelpad=10)
#
# plt.title(args.plot_title, fontsize=20, pad=10.0)
#
# plt.tick_params(labelsize=20)
#
# plt.legend()
#
# fig_path = os.path.join(os.path.dirname(os.getcwd()), 'plots', args.plot_name + '.png')
#
# plt.savefig(fig_path, dpi=200)
#
# plt.close(figure)