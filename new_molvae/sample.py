import os
import sys
import argparse

import torch

import rdkit

# implicitly set PYTHONPATH for the relevant imports
PARENT_DIR_PATH = os.path.abspath('..')
sys.path.insert(0, PARENT_DIR_PATH)

from jtnn import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--depthT', type=int, default=1)
parser.add_argument('--depthG', type=int, default=3)
parser.add_argument('--use_graph_conv', action='store_true')

args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = ClusterVocab(vocab)

model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG, args.num_layers, args.use_graph_conv)
model.load_state_dict(torch.load(args.model))
model = model.cuda()

torch.manual_seed(0)

smiles = []

if args.use_graph_conv:
    for i in range(args.nsample):
        smiles_str = model.sample_prior_graph_conv()
        smiles.append(smiles_str)
        print(smiles_str)

else:
    for i in range(args.nsample):
        smiles_str = model.sample_prior()
        smiles.append(smiles_str)
        print(smiles_str)

validity = len(smiles) / args.nsample

print('Validity: {}'.format(validity))