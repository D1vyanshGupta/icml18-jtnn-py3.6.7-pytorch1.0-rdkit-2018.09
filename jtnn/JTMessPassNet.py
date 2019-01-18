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

def one_hot_encode(x, allowable_set):
    """
    Description: This function, given a categorical variable,
    returns the corresponding one-hot encoding vector.

    Args:
        x: object
            The categorical variable to be one-hot encoded.

        allowable_set: List[object]
            List of all categorical variables in consideration.

    Returns:
         List[Boolean]
            The corresponding one-hot encoding vector.

    """

    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_feature_vec(atom):
    """
    Description: This function, constructs the feature vector for the given atom.

    Args: (object: rdkit)
        The atom for which the feature vector is to be constructed.

    Returns:
        torch.tensor (shape: ATOM_FEATURE_DIM)
    """
    return torch.Tensor(
        # one-hot encode atom symbol / element type
        one_hot_encode(atom.GetSymbol(), ELEM_LIST)
        # one-hot encode atom degree
        + one_hot_encode(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        # one-hot encode formal charge of the atom
        + one_hot_encode(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        # one-hot encoding / binary encoding whether atom is aromatic or not
        + [atom.GetIsAromatic()])

def get_bond_feature_vec(bond):
    """
    Description: This function, constructs the feature vector for the given bond.

    Args:
        bond: (object: rdkit)
            The bond for which the feature vector is to be constructed.

    Returns:
        torch.tensor (shape: BOND_FEATURE_DIM)
    """

    # obtain the bond-type
    bt = bond.GetBondType()
    # one-hot encoding the bond-type
    return torch.Tensor(
        [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
         bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()])

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

    def forward(self, atom_feature_matrix, bond_feature_matrix, atom_adjacency_graph, bond_adjacency_graph, scope, tree_mess):
        """
        Description: Implements the forward pass for encoding the candidate molecular subgraphs, for the graph decoding step. (Section 2.5)

        Args:
            atom_feature_matrix: torch.tensor (shape: num_atoms x ATOM_FEATURE_DIM)
                Matrix of atom features for all the atoms, over all the molecules, in the dataset.

            bond_feature_matrix: torch.tensor (shape: num_bonds x ATOM_FEATURE_DIM + BOND_FEATURE_DIM)
                Matrix of bond features for all the bond, over all the molecules, in the dataset.

            atom_adjacency_graph: torch.tensor (shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For every atom across the training dataset, this atom_graph gives the bond idxs of all the bonds
                in which it is present. An atom can at most be present in MAX_NUM_NEIGHBORS(= 6) bonds.

            bond_adjacency_graph: torch.tensor (shape: num_bonds x MAX_NUM_NEIGHBORS(=6))
                For every non-ring bond (cluster-node) across the training dataset, this bond_graph gives the bond idx of
                those non-ring bonds (cluster-nodes), to which it is connected in the "cluster-graph".

            scope: List[Tuple(int, int)]
                List of tuples of (total_atoms, num_atoms). Used to extract the atom features for a
                particular molecule in the dataset, from the atom_feature_matrix.

        Returns:
            mol_vecs: torch.tensor (shape: num_candidate_subgraphs x hidden_size)
                The encoding of all the candidate subgraphs for scoring purposes. (Section 2.5)

        """
        # create PyTorch Variables
        atom_feature_matrix = create_var(atom_feature_matrix)
        bond_feature_matrix = create_var(bond_feature_matrix)
        atom_adjacency_graph = create_var(atom_adjacency_graph)
        bond_adjacency_graph = create_var(bond_adjacency_graph)

        static_messages = self.W_i(bond_feature_matrix)

        # apply ReLU activation for timestep, t = 0
        graph_message = nn.ReLU()(static_messages)

        # implement message passing for timesteps, t = 1 to T (depth)
        for timestep in range(self.depth - 1):

            message = torch.cat([tree_mess, graph_message], dim=0)

            # obtain messages from all the "inward edges"
            neighbor_message_vecs = index_select_ND(message, 0, bond_adjacency_graph)

            # sum up all the "inward edge" message vectors
            neighbor_message_vecs_sum = neighbor_message_vecs.sum(dim=1)

            # multiply with the weight matrix for the hidden layer
            neighbor_message = self.W_h(neighbor_message_vecs_sum)

            # message at timestep t + 1
            graph_message = nn.ReLU()(static_messages + neighbor_message)

        # neighbor message vectors for each node from the message matrix
        message = torch.cat([tree_mess, graph_message], dim=0)

        # neighbor message for each atom
        neighbor_message_vecs = index_select_ND(message, 0, atom_adjacency_graph)

        # neighbor message for each atom
        neighbor_message_atom_matrix = neighbor_message_vecs.sum(dim=1)

        # concatenate atom feature vector and neighbor hidden message vector
        atom_input_matrix = torch.cat([atom_feature_matrix, neighbor_message_atom_matrix ], dim=1)

        atom_hidden_layer_synaptic_input = nn.ReLU()(self.W_o(atom_input_matrix))

        # list to store the corresponding molecule vectors for each molecule
        mol_vecs = []
        for start_idx, len in scope:
            # mol_vec = atom_hidden_layer_synaptic_input.narrow(0, start_idx, len).sum(dim=0) / len
            mol_vec = atom_hidden_layer_synaptic_input[start_idx: start_idx + len].mean(dim=0)
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

    @staticmethod
    def tensorize(candidate_batch, mess_dict, cuda_device):
        """
        Args:
            candidate_batch: List[Tuple(str, List[MolJuncTreeNode], object: rdkit)]
                The list of candidate subgraphs to be scored, for the graph decoding step (Section 2.5).

            mess_dict: Dict{(MolJuncTreeNode, MolJuncTreeNode): torch.tensor (shape: hidden_size)}
                The dictionary containing edge messages from the tree-encoding step (Section 2.3 and Section 2.5 magic)

        Returns:
            atom_feature_matrix: torch.tensor (shape: num_atoms x ATOM_FEATURE_DIM)
                Matrix of atom features for all the atoms, over all the molecules, in the dataset.

            bond_feature_matrix: torch.tensor (shape: num_bonds x ATOM_FEATURE_DIM + BOND_FEATURE_DIM)
                Matrix of bond features for all the bond, over all the molecules, in the dataset.

            atom_adjacency_graph: torch.tensor (shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For every atom across the training dataset, this atom_graph gives the bond idxs of all the bonds
                in which it is present. An atom can at most be present in MAX_NUM_NEIGHBORS(= 6) bonds.

            bond_adjacency_graph: torch.tensor (shape: num_bonds x MAX_NUM_NEIGHBORS(=6))
                For every non-ring bond (cluster-node) across the training dataset, this bond_graph gives the bond idx of
                those non-ring bonds (cluster-nodes), to which it is connected in the "cluster-graph".

            scope: List[Tuple(int, int)]
                List of tuples of (total_atoms, num_atoms). Used to extract the atom features for a
                particular molecule in the dataset, from the atom_feature_matrix.
        """

        # lists to store atom and bond feature vectors of candidate subgraphs to be encoded by message passing (Section 2.5)
        atom_feature_vecs, bond_feature_vecs = [], []

        # in bonds: for each atom, the list of idx of all bonds, in which it is the terminal atom
        # all_bonds: to store all the bonds (Tuple(MolJuncTreeNode, MolJuncTreeNode)), for all molecules, across the entire dataset.
        in_bonds, all_bonds = [], []

        # for each atom, of every molecule, across the dataset, we give it an idx
        total_atoms = 0

        # the tensor at the 0th index is the padding vector
        total_mess = len(mess_dict) + 1

        # to store tuples of (start_idx, len) to delinate atom features for a particular molecule.
        scope = []

        for smiles, all_nodes, ctr_node in candidate_batch:
            # obtain the rdkit molecule object
            mol = Chem.MolFromSmiles(smiles)

            # obtain the kekulized representation of the molecule object.
            # the original jtnn version kekulizes. Need to revisit as to why it is necessary
            Chem.Kekulize(mol)

            # number of atoms in this molecule, (for scope tuple)
            num_atoms = mol.GetNumAtoms()

            for atom in mol.GetAtoms():
                # obtain the feature vector for the given atom
                atom_feature_vecs.append(get_atom_feature_vec(atom))

                # append an empty list of every molecule, to store idxs of all the bonds, in which it is the terminal atom.
                in_bonds.append([])

            for bond in mol.GetBonds():
                bond_begin_idx = bond.GetBeginAtom()
                bond_end_idx = bond.GetEndAtom()

                # offsetted begin and end atom idxs
                x = bond_begin_idx.GetIdx() + total_atoms
                y = bond_end_idx.GetIdx() + total_atoms

                # Here x_nid,y_nid could be 0

                # retrieve the idxs of the nodes in the junction tree, in whose corresponding clusters, the atoms are included
                x_nid, y_nid = bond_begin_idx.GetAtomMapNum(), bond_end_idx.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                # obtain the feature vector for this bond.
                bond_feature_vec = get_bond_feature_vec(bond)

                # bond idx offseted by total_mess
                bond_offset_idx = total_mess + len(all_bonds)
                all_bonds.append((x, y))
                bond_feature_vecs.append(torch.cat([atom_feature_vecs[x], bond_feature_vec], 0))
                in_bonds[y].append(bond_offset_idx)

                bond_offset_idx = total_mess + len(all_bonds)
                all_bonds.append((y, x))
                bond_feature_vecs.append(torch.cat([atom_feature_vecs[y], bond_feature_vec], 0))
                in_bonds[x].append(bond_offset_idx)

                # the weird message passing magic in (Section 2.5)
                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid, y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid, y_bid)]
                        in_bonds[y].append(mess_idx)
                    if (y_bid, x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid, x_bid)]
                        in_bonds[x].append(mess_idx)

            # append start_idx and len for delinating atom features for a particular molecule
            scope.append((total_atoms, num_atoms))
            total_atoms += num_atoms

        total_bonds = len(all_bonds)
        atom_feature_matrix = torch.stack(atom_feature_vecs, 0)
        bond_feature_matrix = torch.stack(bond_feature_vecs, 0)
        atom_adjacency_graph = torch.zeros(total_atoms, MAX_NUM_NEIGHBORS).long()
        bond_adjacency_graph = torch.zeros(total_bonds, MAX_NUM_NEIGHBORS).long()

        # for each atoms, the list of idxs of all the bonds, in which it is the terminal atom
        for a in range(total_atoms):
            for i, bond_offset_idx in enumerate(in_bonds[a]):
                atom_adjacency_graph[a, i] = bond_offset_idx

        # for each bond, the list of idx of "inward" "bonds", for message passing purposes
        for b1 in range(total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]):
                # b2 is offseted by total_mess
                if b2 < total_mess or all_bonds[b2 - total_mess][0] != y:
                    bond_adjacency_graph[b1, i] = b2

        atom_feature_matrix.to(cuda_device)
        bond_feature_matrix.to(cuda_device)
        atom_adjacency_graph.to(cuda_device)
        bond_adjacency_graph.to(cuda_device)

        return (atom_feature_matrix, bond_feature_matrix, atom_adjacency_graph, bond_adjacency_graph, scope)