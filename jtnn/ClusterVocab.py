import copy
import rdkit
from MolJuncTree import MolJuncTree
import rdkit.Chem as Chem


def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]


class ClusterVocab(object):
    """
    This class takes a list of SMILES strings,
    that correspond to the cluster vocabulary of the training dataset.
    """

    # class constructor
    #
    def __init__(self, smiles_list):
        """
        This is the constructor for the ClusterVocab class.

        Args:
            smiles_list: list of SMILES representations, that correspond to the cluster vocabulary over the training dataset.

        Returns:
            ClusterVocab object for the corresponding training dataset.
        """

        # list of SMILES representations, corresponding to the cluster vocabulary
        self.smiles_vocab = smiles_list

        self.vocab_map = {smiles: idx for idx, smiles in enumerate(self.smiles_vocab)}

        self.slots = [get_slots(smiles) for smiles in self.smiles_vocab]
    # given SMILES string, corresponding to a cluster,
    # returns its index in the vocabulary map
    def get_index(self, smiles):
        """
        This method gives the index corresponding to the given cluster vocabulary item.

        Args:
            smiles: SMILES representaion of a cluster vocabulary item.

        Returns:
             Index of the corresponding cluster vocabulary item.
        """
        return self.vocab_map[smiles]

    # given an index, return the corresponding SMILES string
    def get_smiles(self, idx):
        """
        This method returns the corresponding the SMILES representation for the cluster vocabulary item, given an index.

        Args:
            idx: index of the cluster vocabulary item

        Returns:
             The SMILES representation of the corresponding cluster vocabulary item.
        """
        return self.smiles_vocab[idx]

    def size(self):
        return len(self.smiles_vocab)

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

# if __name__ == "__main__":
#     import sys
#     import os
#     lg = rdkit.RDLogger.logger()
#     lg.setLevel(rdkit.RDLogger.CRITICAL)
#
#     smiles_set = set()
#     for idx, line in enumerate(sys.stdin):
#         smiles = line.split()[0]
#         junc_tree = MolJuncTree(smiles)
#         for node in junc_tree.nodes:
#             smiles_set.add(node.smiles)
#     for smiles in smiles_set:
#         print(smiles)
#     WRITE_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data', 'vocab_50.txt')
#     with open(WRITE_PATH, 'w') as file:
#         for smiles in smiles_set:
#             file.write(smiles + "\n")
