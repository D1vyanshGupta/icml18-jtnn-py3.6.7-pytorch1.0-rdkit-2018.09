import os, random
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from JTNNEncoder import JTNNEncoder
from MessPassNet import MessPassNet
from JTMessPassNet import JTMessPassNet
from MolGraphEncoder import MolGraphEncoder

# class PairTreeFolder(object):
#
#     def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
#         self.data_folder = data_folder
#         self.data_files = [fn for fn in os.listdir(data_folder)]
#         self.batch_size = batch_size
#         self.vocab = vocab
#         self.num_workers = num_workers
#         self.y_assm = y_assm
#         self.shuffle = shuffle
#
#         if replicate is not None: #expand is int
#             self.data_files = self.data_files * replicate
#
#     def __iter__(self):
#         for fn in self.data_files:
#             fn = os.path.join(self.data_folder, fn)
#             with open(fn) as f:
#                 data = pickle.load(f)
#
#             if self.shuffle:
#                 random.shuffle(data) #shuffle data before batch
#
#             batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
#             if len(batches[-1]) < self.batch_size:
#                 batches.pop()
#
#             dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
#             dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])
#
#             for b in dataloader:
#                 yield b
#
#             del data, batches, dataset, dataloader

class MolTreeFolder(object):

    def __init__(self, data_folder, vocab, use_graph_conv, batch_size, cuda_device, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.cuda_device = cuda_device
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
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.use_graph_conv, self.cuda_device, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            # free up memory
            del data, batches, dataset, dataloader


# class PairTreeDataset(Dataset):
#
#     def __init__(self, data, vocab, y_assm):
#         self.data = data
#         self.vocab = vocab
#         self.y_assm = y_assm
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         batch0, batch1 = zip(*self.data[idx])
#         return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)


class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, use_graph_conv, cuda_device, assm=True):
        self.data = data
        self.vocab = vocab
        self.use_graph_conv = use_graph_conv
        self.cuda_device = cuda_device
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, self.use_graph_conv, self.cuda_device, assm=self.assm)

def tensorize(junc_tree_batch, vocab, use_graph_conv, cuda_device, assm=True):
    set_batch_nodeID(junc_tree_batch, vocab)
    smiles_batch = [junc_tree.smiles for junc_tree in junc_tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(junc_tree_batch, cuda_device)

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

        stereo_candidates = []
        stereo_batch_idx = []
        stereo_labels = []
        for idx, junc_tree in enumerate(junc_tree_batch):
            candidates = junc_tree.stereo_candidates
            if len(candidates) == 1:
                continue
            if junc_tree.smiles3D not in candidates:
                candidates.append(junc_tree.smiles3D)

            stereo_candidates.extend(candidates)
            stereo_batch_idx.extend([idx] * len(candidates))
            stereo_labels.append( (candidates.index(junc_tree.smiles3D), len(candidates)) )

        stereo_molenc_holder = None
        if len(stereo_labels) > 0:
            stereo_molenc_holder = MolGraphEncoder.tensorize(stereo_candidates)
        stereo_batch_idx = torch.LongTensor(stereo_batch_idx)

        return junc_tree_batch, jtenc_holder, molenc_holder, (cand_molenc_holder, cand_batch_idx), (stereo_molenc_holder, stereo_batch_idx, stereo_labels)

    else:
        mpn_holder = MessPassNet.tensorize(smiles_batch, cuda_device)

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

        jtmpn_holder = JTMessPassNet.tensorize(candidates, mess_dict, cuda_device)
        cand_batch_idx = torch.LongTensor(cand_batch_idx)

        stereo_candidates = []
        stereo_batch_idx = []
        stereo_labels = []
        for idx, junc_tree in enumerate(junc_tree_batch):
            candidates = junc_tree.stereo_candidates
            if len(candidates) == 1:
                continue
            if junc_tree.smiles3D not in candidates:
                candidates.append(junc_tree.smiles3D)

            stereo_candidates.extend(candidates)
            stereo_batch_idx.extend([idx] * len(candidates))
            stereo_labels.append((candidates.index(junc_tree.smiles3D), len(candidates)))

        stereo_molenc_holder = None
        if len(stereo_labels) > 0:
            stereo_molenc_holder = MessPassNet.tensorize(stereo_candidates, cuda_device)
        stereo_batch_idx = torch.LongTensor(stereo_batch_idx)

        stereo_batch_idx.to(stereo_batch_idx)

        return junc_tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, cand_batch_idx), (stereo_molenc_holder, stereo_batch_idx, stereo_labels)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1