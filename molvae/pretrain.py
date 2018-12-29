import os, sys
PARENT_DIR_PATH = os.path.abspath('..')
sys.path.insert(0, PARENT_DIR_PATH)

from optparse import OptionParser

import rdkit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from jtnn import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

# parser = OptionParser()
# parser.add_option("-t", "--train", dest="train_path")
# parser.add_option("-v", "--vocab", dest="vocab_path")
# parser.add_option("-s", "--save_dir", dest="save_path")
# parser.add_option("-b", "--batch", dest="batch_size", default=40)
# parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
# parser.add_option("-l", "--latent", dest="latent_size", default=56)
# parser.add_option("-d", "--depth", dest="depth", default=3)
# opts, args = parser.parse_args()

VOCAB_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data', 'vocab_50.txt')
TRAIN_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data', 'train_50.txt')
SAVE_PATH = os.path.join(os.getcwd(), 'pre_model')

# vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = [x.strip("\r\n ") for x in open(VOCAB_PATH)]
vocab = ClusterVocab(vocab)

# batch_size = int(opts.batch_size)
# hidden_size = int(opts.hidden_size)
# latent_size = int(opts.latent_size)
# depth = int(opts.depth)

batch_size = int(5)
hidden_size = int(450)
latent_size = int(56)
depth = int(3)

model = JTNNVAE(vocab, hidden_size, latent_size, depth)

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

# model = model.cuda()
# print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

# dataset = MoleculeDataset(opts.train_path)
dataset = MoleculeDataset(TRAIN_PATH)

MAX_EPOCH = 3
PRINT_ITER = 20

for epoch in range(MAX_EPOCH):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: x,
                            drop_last=True)

    word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0

    for it, batch in enumerate(dataloader):
        for mol_tree in batch:
            for node in mol_tree.nodes:
                if node.label not in node.candidates:
                    node.candidates.append(node.label)
                    node.candidate_mols.append(node.label_mol)

        model.zero_grad()
        loss, kl_div, wacc, tacc, sacc, dacc = model(batch, beta=0)
        print(loss.item(), kl_div, wacc, tacc, sacc, dacc)
        loss.backward()
        optimizer.step()

        word_acc += wacc
        topo_acc += tacc
        assm_acc += sacc
        steo_acc += dacc

        if (it + 1) % PRINT_ITER == 0:
            word_acc = word_acc / PRINT_ITER * 100
            topo_acc = topo_acc / PRINT_ITER * 100
            assm_acc = assm_acc / PRINT_ITER * 100
            steo_acc = steo_acc / PRINT_ITER * 100

            print("KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f" % (kl_div, word_acc, topo_acc, assm_acc, steo_acc))

            word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0
            sys.stdout.flush()

    scheduler.step()
    print("learning rate: %.6f" % scheduler.get_lr()[0])
    # torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))
    torch.save(model.state_dict(), SAVE_PATH + "/model.iter-" + str(epoch))