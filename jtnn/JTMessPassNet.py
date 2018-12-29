import torch
import torch.nn as nn
from nnutils import create_var, index_select_ND
import rdkit.Chem as Chem

# list of elements are were are considering in our problem domain
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

# dimension of the atom feature vector
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1

# one-hot encoding for the atom / element type
# one-hot encoding for the degree of the atom ([0, 1, 2, 3, 4, 5])
# one-hot encoding for the formal charge of the atom ([-2, -1, 0, 1, 2])
# one-hot encoding / binary encoding whether atom is aromatic or not
ATOM_FEATURE_DIM = len(ELEM_LIST) + 6 + 5 + 1

# dimension of the bond feature vector
BOND_FDIM = 5

# one-hot encoding for the bond-type ([single-bond, double-bond, triple-bond, aromatic-bond, in-ring])
BOND_FEATURE_DIM = 5

# maximum number of neighbors of atom in molecule
MAX_NUM_NEIGHBORS = 10

class JTMessPassNet(nn.Module):

    def __init__(self, hidden_size, depth):
        """
        Constructor for the class.

        Args:
            hidden_size: Dimension of the hidden message vectors.
            depth: Number of timesteps for message passing.
        """

        # invoke class constructor
        super(JTMessPassNet, self).__init__()

        # dimension of the hidden message vectors
        self.hidden_size = hidden_size

        # number of timesteps for message passing
        self.depth = depth

        # weight matrices
        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, candidate_batch, tree_mess):
        """
        Args:
            candidate_batch: Batch / list of tuples for all nodes, of all junction trees, across the entire training dataset.
            tree_mess: Dictionary that stores hidden messages calculated in the tree encoding step.

        Returns:

        """
        atom_feature_vecs, bond_feature_vecs = [], []

        in_bonds, all_bonds = [], []

        mess_dict, all_mess = {}, [create_var(torch.zeros(self.hidden_size))]

        total_atoms = 0

        scope = []

        for edge_tuple, message_vec in tree_mess.items():
            mess_dict[edge_tuple] = len(all_mess)
            all_mess.append(message_vec)

        # across all nodes,
        # all molecules,
        # entire training dataset
        for mol, all_nodes, ctr_node in candidate_batch:
            num_atoms = mol.GetNumAtoms()

            for atom in mol.GetAtoms():
                atom_feature_vecs.append(self.get_atom_feature_vec(atom))
                in_bonds.append([])

            for bond in mol.GetBonds():
                bond_begin_atom = bond.GetBeginAtom()
                bond_end_atom = bond.GetEndAtom()

                x = bond_begin_atom.GetIdx() + total_atoms
                y = bond_end_atom.GetIdx() + total_atoms

                # here x_nid, y_nid could be 0
                x_nid, y_nid = bond_begin_atom.GetAtomMapNum(), bond_end_atom.GetAtomMapNum()

                # get node_id of the node, which corresponds to that cluster, to which the atoms belong
                # the alpha_v magic in section 2.5
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                bond_feature_vec = self.get_bond_feature_vec(bond)

                # bond idx offseted by len(all_mess)
                bond_idx = len(all_mess) + len(all_bonds)
                all_bonds.append((x, y))
                bond_feature_vecs.append(torch.cat([atom_feature_vecs[x], bond_feature_vec], 0))
                in_bonds[y].append(bond_idx)

                bond_idx = len(all_mess) + len(all_bonds)
                all_bonds.append((y, x))
                bond_feature_vecs.append(torch.cat([atom_feature_vecs[y], bond_feature_vec], 0))
                in_bonds[x].append(bond_idx)

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid, y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid, y_bid)]
                        in_bonds[y].append(mess_idx)
                    if (y_bid, x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid, x_bid)]
                        in_bonds[x].append(mess_idx)

            scope.append((total_atoms, num_atoms))
            total_atoms += num_atoms

        total_bonds = len(all_bonds)
        total_mess = len(all_mess)
        atom_feature_vecs = torch.stack(atom_feature_vecs, 0)
        bond_feature_vecs = torch.stack(bond_feature_vecs, 0)
        atom_graph = torch.zeros(total_atoms, MAX_NUM_NEIGHBORS).long()
        bond_graph = torch.zeros(total_bonds, MAX_NUM_NEIGHBORS).long()
        tree_message = torch.stack(all_mess, dim=0)

        for atom_idx in range(total_atoms):
            for neighbor_idx, bond_idx in enumerate(in_bonds[atom_idx]):
                atom_graph[atom_idx, neighbor_idx] = bond_idx

        for bond_idx_1 in range(total_bonds):
            x, y = all_bonds[bond_idx_1]
            for neighbor_idx, bond_idx_2 in enumerate(in_bonds[x]):
                # b2 is offseted by len(all_mess)
                if bond_idx_2 < total_mess or all_bonds[bond_idx_2 - total_mess][0] != y:
                    bond_graph[bond_idx_1, neighbor_idx] = bond_idx_2

        atom_feature_vecs = create_var(atom_feature_vecs)
        bond_feature_vecs = create_var(bond_feature_vecs)
        atom_graph = create_var(atom_graph)
        bond_graph = create_var(bond_graph)

        bond_features_synaptic_input = self.W_i(bond_feature_vecs)
        graph_message = nn.ReLU()(bond_features_synaptic_input)

        for idx in range(self.depth - 1):
            message = torch.cat([tree_message, graph_message], dim=0)
            neighbor_message = index_select_ND(message, 0, bond_graph)
            neighbor_message = neighbor_message.sum(dim=1)
            neighbor_message = self.W_h(neighbor_message)
            graph_message = nn.ReLU()(bond_features_synaptic_input + neighbor_message)

        message = torch.cat([tree_message, graph_message], dim=0)
        neighbor_message = index_select_ND(message, 0, atom_graph)
        neighbor_message = neighbor_message.sum(dim=1)
        atom_input = torch.cat([atom_feature_vecs, neighbor_message], dim=1)
        atom_hiddens = nn.ReLU()(self.W_o(atom_input))

        mol_vecs = []
        for st, le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

    def one_hot_encode(self, x, allowable_set):
        """
        This function, given a categorical variable,
        returns the corresponding one-hot encoding vector.

        Args:
            x: Categorical variables.
            allowable_set: List of all categorical variables in consideration.

        Returns:
             The corresponding one-hot encoding vector.
        """

        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def get_atom_feature_vec(self, atom):
        """
        This function, constructs the feature vector for the given atom.
        """
        return torch.Tensor(
            # one-hot encode atom symbol / element type
            self.one_hot_encode(atom.GetSymbol(), ELEM_LIST)
            # one-hot encode atom degree
            + self.one_hot_encode(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            # one-hot encode formal charge of the atom
            + self.one_hot_encode(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
            # one-hot encoding / binary encoding whether atom is aromatic or not
            + [atom.GetIsAromatic()])

    def get_bond_feature_vec(self, bond):
        """
        This function, constructs the feature vector for the given bond.
        """

        # obtain the bond-type
        bt = bond.GetBondType()
        # one-hot encoding the bond-type
        return torch.Tensor(
            [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
             bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()])
