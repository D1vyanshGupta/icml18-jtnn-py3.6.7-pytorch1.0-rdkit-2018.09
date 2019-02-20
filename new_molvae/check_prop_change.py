import os
import sys
import argparse

from timeit import default_timer as timer

import rdkit
from rdkit import Chem

from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles

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
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)
parser.add_argument('--use_graph_conv', action='store_true')

args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = ClusterVocab(vocab)

model = JTNNVAE_Prop(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG, args.num_layers, args.use_graph_conv, args.training_prop_pred)
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

prop_val_change = 0

postive_increase = 0
postive_increase_cnt = 0

negative_decrease = 0
negative_decrease_cnt = 0

if args.use_graph_conv:
    for smiles in data:
        old_prop_val = Descriptors.MolLogP(MolFromSmiles(smiles))

        try:
            dec_smiles = model.reconstruct_graph_conv(smiles)
            new_prop_val = Descriptors.MolLogP(MolFromSmiles(dec_smiles))

            prop_val_change = new_prop_val - old_prop_val
            tot += 1

            if new_prop_val >= old_prop_val:
                postive_increase = new_prop_val - old_prop_val
                postive_increase_cnt += 1

            else:
                negative_decrease = new_prop_val - old_prop_val
                negative_decrease_cnt += 1

        except Exception as e:
            print(e)

        print(tot)

    end_time = timer()

    total_time = (end_time - start_time)

    print('Time taken: {}'.format(total_time / 60))
    print('Net Average Change: {}'.format(prop_val_change/ tot))
    print('Positive Increase for {} out of {}. Net Average Positive Increase: {}'.format(postive_increase_cnt, tot, postive_increase / postive_increase_cnt))
    print('Negative Decrease for {} out of {}. Net Average Positive Increase: {}'.format(negative_decrease_cnt, tot, negative_decrease / negative_decrease_cnt))

else:
    for smiles in data:
        old_prop_val = Descriptors.MolLogP(MolFromSmiles(smiles))

        try:
            dec_smiles = model.reconstruct(smiles)
            new_prop_val = Descriptors.MolLogP(MolFromSmiles(dec_smiles))

            prop_val_change = new_prop_val - old_prop_val
            tot += 1

            if new_prop_val >= old_prop_val:
                postive_increase = new_prop_val - old_prop_val
                postive_increase_cnt += 1

            else:
                negative_decrease = new_prop_val - old_prop_val
                negative_decrease_cnt += 1

        except Exception as e:
            print(e)

        print(tot)

    end_time = timer()

    total_time = (end_time - start_time)

    print('Time taken: {}'.format(total_time / 60))
    print('Net Average Change: {}'.format(prop_val_change/ tot))
    print('Positive Increase for {} out of {}. Net Average Positive Increase: {}'.format(postive_increase_cnt, tot, postive_increase / postive_increase_cnt))
    print('Negative Decrease for {} out of {}. Net Average Positive Increase: {}'.format(negative_decrease_cnt, tot, negative_decrease / negative_decrease_cnt))
