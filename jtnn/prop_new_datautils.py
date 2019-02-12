import os, random
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from JTNNEncoder import JTNNEncoder
from MessPassNet import MessPassNet
from JTMessPassNet import JTMessPassNet
from MolGraphEncoder import MolGraphEncoder

from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles

class PropMolTreeFolder(object):

    def __init__(self, data_folder, vocab, use_graph_conv, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.use_graph_conv = use_graph_conv
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        # expand is int
        if replicate is not None:
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle:
                # shuffle data before batch
                random.shuffle(data)

            batches = [data[idx : idx + self.batch_size] for idx in range(0, len(data), self.batch_size)]

            dataset = MolTreeDataset(batches, self.vocab, self.use_graph_conv, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            # free up memory
            del data, batches, dataset, dataloader

class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, use_graph_conv, assm=True):
        self.data = data
        self.vocab = vocab
        self.use_graph_conv = use_graph_conv
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, self.use_graph_conv, assm=self.assm)

def tensorize(junc_tree_batch, vocab, use_graph_conv, assm=True):
    set_batch_nodeID(junc_tree_batch, vocab)
    smiles_batch = [junc_tree.smiles for junc_tree in junc_tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(junc_tree_batch)

    prop_batch = []
    for smiles in smiles_batch:
        prop_batch.append(Descriptors.MolLogP(MolFromSmiles(smiles)))

    if use_graph_conv:
        molenc_holder = MolGraphEncoder.tensorize(smiles_batch)

        if assm is False:
            return junc_tree_batch, jtenc_holder, molenc_holder

        candidate_smiles = []
        cand_batch_idx = []
        for idx, junc_tree in enumerate(junc_tree_batch):
            for node in junc_tree.nodes:
                # leaf node's attachment is determined by neighboring node's attachment
                if node.is_leaf or len(node.candidates) == 1:
                    continue
                candidate_smiles.extend([candidate for candidate in node.candidates])
                cand_batch_idx.extend([idx] * len(node.candidates))

        cand_molenc_holder = MolGraphEncoder.tensorize(candidate_smiles)
        cand_batch_idx = torch.LongTensor(cand_batch_idx)

        return junc_tree_batch, jtenc_holder, molenc_holder, (cand_molenc_holder, cand_batch_idx)

    else:
        mpn_holder = MessPassNet.tensorize(smiles_batch)

        if assm is False:
            return junc_tree_batch, jtenc_holder, mpn_holder

        candidates = []
        cand_batch_idx = []
        for idx, junc_tree in enumerate(junc_tree_batch):
            for node in junc_tree.nodes:
                # leaf node's attachment is determined by neighboring node's attachment
                if node.is_leaf or len(node.candidates) == 1:
                    continue
                candidates.extend([(candidate, junc_tree.nodes, node) for candidate in node.candidates])
                cand_batch_idx.extend([idx] * len(node.candidates))

        jtmpn_holder = JTMessPassNet.tensorize(candidates, mess_dict)
        cand_batch_idx = torch.LongTensor(cand_batch_idx)

        return junc_tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, cand_batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1