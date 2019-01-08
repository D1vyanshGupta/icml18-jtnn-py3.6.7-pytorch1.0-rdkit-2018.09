import torch
from torch.utils.data import Dataset
from MolJuncTree import MolJuncTree
import numpy as np


class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n").split()[0] for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        # mol_tree = MolJuncTree(smiles)
        # mol_tree.recover()
        # mol_tree.assemble()
        ascii_char_list = [ord(ch) for ch in smiles]
        np_arr = np.array(ascii_char_list)
        torch_tensor = torch.from_numpy(np_arr)
        return torch_tensor


class PropDataset(Dataset):

    def __init__(self, data_file, prop_file):
        self.prop_data = np.loadtxt(prop_file)
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolJuncTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree, self.prop_data[idx]