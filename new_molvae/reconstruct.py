import os
import sys
import argparse

from timeit import default_timer as timer

import torch

import rdkit
from rdkit import Chem

# implicitly set PYTHONPATH for the relevant imports
PARENT_DIR_PATH = os.path.abspath('..')
sys.path.insert(0, PARENT_DIR_PATH)

from jtnn import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--test', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--hidden_size', type=int, default=450)
# parser.add_argument('--hidden_size', type=int, default=150)
parser.add_argument('--latent_size', type=int, default=56)
# parser.add_argument('--latent_size', type=int, default=20)
parser.add_argument('--num_layers', type=int, default=2)
# parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthT', type=int, default=1)
parser.add_argument('--depthG', type=int, default=3)
parser.add_argument('--use_graph_conv', action='store_true')

args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = ClusterVocab(vocab)

model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG, args.num_layers, args.use_graph_conv)
model.load_state_dict(torch.load(args.model))
model = model.cuda()

data = []
with open(args.test) as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

acc = 0.0
tot = 0
start_time = timer()

if args.use_graph_conv:
    for smiles in data:
        mol = Chem.MolFromSmiles(smiles)
        smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)

        try:
            dec_smiles = model.reconstruct_graph_conv(smiles3D)

            if dec_smiles == smiles3D:
                acc += 1
        except Exception as e:
            print(e)
        tot += 1
        print(tot)

    end_time = timer()

    total_time = (end_time - start_time)
    print('Reconstruction Test Accuracy: {}. Time taken: {}'.format(acc / tot, total_time / 60))

else:
    for smiles in data:
        mol = Chem.MolFromSmiles(smiles)
        smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)

        try:
            dec_smiles = model.reconstruct(smiles3D)

            if dec_smiles == smiles3D:
                acc += 1
        except Exception as e:
            print(e)
        tot += 1
        print(tot)

    end_time = timer()

    total_time = (end_time - start_time)
    print('Reconstruction Test Accuracy: {}. Time taken: {}'.format(acc / tot, total_time / 60))

"""
dec_smiles = model.recon_eval(smiles3D)
tot += len(dec_smiles)
for s in dec_smiles:
    if s == smiles3D:
        acc += 1
    print acc / tot
    """