import os
import sys
import argparse

import numpy as np

import torch

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import sascorer

# implicitly set PYTHONPATH for the relevant imports
PARENT_DIR_PATH = os.path.abspath('..')
sys.path.insert(0, PARENT_DIR_PATH)
from jtnn import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--hidden_size', type=int, default=450)
# parser.add_argument('--hidden_size', type=int, default=200)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)
parser.add_argument('--use_graph_conv', action='store_true')

args = parser.parse_args()

with open(args.data) as f:
    smiles = f.readlines()

for i in range(len(smiles)):
    smiles[i] = smiles[i].strip()

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = ClusterVocab(vocab)

batch_size = 100
# hidden_size = int(args.hidden_size)
# latent_size = int(args.latent_size)
# depth = int(opts.depth)

model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG, args.num_layers, args.use_graph_conv)
model.load_state_dict(torch.load(args.model))
model = model.cuda()

smiles_rdkit = []
for i in range(len(smiles)):
    print(i, 'smiles')
    smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles[i]), isomericSmiles=True))

logP_values = []
for i in range(len(smiles)):
    print(i, 'logP_values')
    logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[i])))

SA_scores = []
for i in range(len(smiles)):
    print(i, 'SA_scores')
    SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[i])))

import networkx as nx

cycle_scores = []
for i in range(len(smiles)):
    print(i, 'cycle_scores')
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[ i ]))))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_scores.append(-cycle_length)

SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

latent_points = []
for i in range(0, len(smiles), batch_size):
    print(i, 'latent points')
    batch = smiles[i:i+batch_size]
    mol_vec = model.encode_latent_mean(batch)
    latent_points.append(mol_vec.data.cpu().numpy())

# We store the results
latent_points = np.vstack(latent_points)
np.savetxt('latent_features.txt', latent_points)

targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
np.savetxt('targets.txt', targets)
np.savetxt('logP_values.txt', np.array(logP_values))
np.savetxt('SA_scores.txt', np.array(SA_scores))
np.savetxt('cycle_scores.txt', np.array(cycle_scores))