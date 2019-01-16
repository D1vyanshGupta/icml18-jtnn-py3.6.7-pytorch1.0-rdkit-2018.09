import torch
import torch.nn as nn
import torch.nn.functional as F

import rdkit.Chem as Chem
from chemutils import get_kekulized_mol_from_smiles
from nnutils import create_var

from ConvNetLayer import ConvNetLayer

# list of elements under consideration in the problem domain
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

# one-hot encoding for the atom / element type
# one-hot encoding for the degree of the atom ([0, 1, 2, 3, 4, 5])
# one-hot encoding for the formal charge of the atom ([-2, -1, 0, 1, 2])
# one-hot enoding for the chiral-tag of the atom i.e. number of chiral centres ([0, 1, 2, 3])
# one-hot encoding / binary encoding whether atom is aromatic or not
ATOM_FEATURE_DIM = len(ELEM_LIST) + 6 + 5 + 4 + 1

# one-hot encoding for the bond-type ([single-bond, double-bond, triple-bond, aromatic-bond, in-ring])
# one-hot encoding for the stereo-configuration of the bond ([0, 1, 2, 3, 4, 5])
BOND_FEATURE_DIM = 5 + 6

# maximum number of an atom in a molecule (Chemistry Domain Knowledge)
MAX_NUM_NEIGHBORS = 6


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
         The corresponding one-hot encoding vector.
    """

    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_feature_vec(atom):
    """
    Description: This method, constructs the feature vector for the given atom.

    Args:
        atom: (object: rdkit)
            The atom (rdkit object) whose feature vector has to be obtained.

    Returns: torch.tensor
        The corresponding feature vector for the given atom.
    """

    return torch.Tensor(
        # one-hot encode atom symbol / element type
        one_hot_encode(atom.GetSymbol(), ELEM_LIST)
        # one-hot encode atom degree
        + one_hot_encode(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        # one-hot encode formal charge of the atom
        + one_hot_encode(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        # one-hot encode the chiral tag of the atom
        + one_hot_encode(int(atom.GetChiralTag()), [0, 1, 2, 3])
        # one-hot encoding / binary encoding whether atom is aromatic or not
        + [atom.GetIsAromatic()]
    )

def get_bond_feature_vec(bond):
    """
    Description: This function, constructs the feature vector for the given bond.

    Args:
        bond: (object: rdkit)
            The bond whose feature vector has to be constructed.

    Returns: torch.tensor
        The corresponding feature vector for the given bond.
    """

    # obtain the bond-type
    bt = bond.GetBondType()
    # obtain the stereo-configuration
    stereo = int(bond.GetStereo())
    # one-hot encoding the bond-type
    bond_type_feature = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                         bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    # one-hot encoding the stereo configuration
    stereo_conf_feature = one_hot_encode(stereo, [0, 1, 2, 3, 4, 5])
    return torch.Tensor(bond_type_feature + stereo_conf_feature)


class MolGraphEncoder(nn.Module):
    """
    Description: This module implements a "Graph Convolutional Neural Network" for encoding molecular graphs.
    """

    def __init__(self, hidden_size, num_layers):
        """
        Description: Constructor for the class.

        Args:

            hidden_size: int
                The dimension of the hidden feature vectors to be used.

            num_layers: int
                The number of "convolutional layers" in the graph convolutional encoder.

            atom_embedding: torch.enbedding
                The embedding space for translating the sparse atom feature vectors,
                to lower (sometimes, maybe higher) dimensional and dense vectors.

            bond_embedding: torch.enbedding
                The embedding space for translating the sparse bond feature vectors,
                to low (sometimes, maybe high) dimensional and dense vectors.
        """

        # invoke the superclass constructor
        super(MolGraphEncoder, self).__init__()

        # the dimension of the hidden feature vectors to be used
        self.hidden_size = hidden_size

        # number of "convolutional layers" in the graph encoder
        self.num_layers = num_layers

        # list to store "convolutional layers" for the encoder
        conv_layers = []

        # instantiate the convnet layers for the encoder
        for layer_idx in range(num_layers):
            # input ATOM_FEATURE_DIM and BOND_FEATURE_DIM, for instantiating the first layer
            if layer_idx == 0:
                # instantiate a convnet layer
                conv_layer = ConvNetLayer(hidden_size, atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM)

            else:
                # instantiate a convnet layer
                conv_layer = ConvNetLayer(hidden_size)

            # append this layer to the list of convnet layers
            conv_layers.append(conv_layer)

        # finally, set convnet layers for the encoder
        self.conv_layers = nn.ModuleList(conv_layers)

        # weight matrix for "edge pooling"
        self.A = nn.Linear(hidden_size, hidden_size, bias=True)

        # weight matrices for "edge gate" computation (as per Prof. Bresson's code)

        # weight matrix for the edge feature
        self.U = nn.Linear(hidden_size, hidden_size, bias=True)

        # weight matrix for feature vector of atom 'i' for edge e_(i, j)
        self.V = nn.Linear(hidden_size, hidden_size, bias=True)

        # weight matrix for feature vector of atom 'j' for edge e_(i, j)
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, atom_feature_matrix, bond_feature_matrix, atom_adjacency_graph, atom_bond_adjacency_graph, bond_atom_adjacency_graph, scope):
        """
        Args:
            atom_feature_matrix: torch.tensor (shape: batch_size x atom_feature_dim)
                The matrix containing feature vectors, for all the atoms, across the entire batch.
                * atom_feature_dim = len(ELEM_LIST) + 6 + 5 + 4 + 1

            bond_feature_matrix: torch.tensor (shape: batch_size x bond_feature_dim)
                The matrix containing feature vectors, for all the bonds, across the entire batch.
                * bond_feature_dim = 5 + 6

            atom_adjacency_graph: torch.tensor (shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For each atom, across the entire batch, the idxs of neighboring atoms.

            atom_bond_adjacency_graph: torch.tensor(shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For each atom, across the entire batch, the idxs of all the bonds, in which it is the initial atom.

            bond_atom_adjacency_graph: torch.tensor (shape: num_bonds x 2)
                For each bond, across the entire batch, the idxs of the 2 atoms, of which the bond is composed of.

            scope: List[Tuple(int, int)]
                The list to store tuples (total_bonds, num_bonds), to keep track of all the bond feature vectors,
                belonging to a particular molecule.

        Returns:
            mol_vecs: torch.tensor (shape: batch_size x hidden_size)
                The hidden vector representation of each molecular graph, across the entire batch
        """
        # create PyTorch variables
        atom_feature_matrix = create_var(atom_feature_matrix)
        bond_feature_matrix = create_var(bond_feature_matrix)
        atom_adjacency_graph = create_var(atom_adjacency_graph)
        atom_bond_adjacency_graph = create_var(atom_bond_adjacency_graph)
        bond_atom_adjacency_graph = create_var(bond_atom_adjacency_graph)

        # implement convolution
        atom_layer_input = atom_feature_matrix
        bond_layer_input = bond_feature_matrix

        for conv_layer in self.conv_layers:
            # implement forward pass for this convolutional layer
            atom_layer_output, bond_layer_output = conv_layer(atom_layer_input, bond_layer_input, atom_adjacency_graph,
                                                              atom_bond_adjacency_graph, bond_atom_adjacency_graph)

            # set the input features for the next convolutional layer
            atom_layer_input, bond_layer_input = atom_layer_output, bond_layer_output

        # for each molecular graph, pool all the edge feature vectors
        mol_vecs = self.pool_bond_features_for_mols(atom_layer_output, bond_layer_output, bond_atom_adjacency_graph, scope)

        return mol_vecs

    def pool_bond_features_for_mols(self, atom_layer_output, bond_layer_output, bond_atom_adjacency_graph, scope):
        """
        Args:
            atom_layer_output: torch.tensor (shape: batch_size x atom_feature_dim)
                The matrix containing feature vectors, for all the atoms, across the entire batch.

            bond_layer_output: torch.tensor (shape: batch_size x bond_feature_dim)
                The matrix containing feature vectors, for all the bonds, across the entire batch.

            bond_atom_adjacency_graph: torch.tensor (shape: num_bonds x 2)

            scope: List[Tuple(int, int)]
                The list to store tuples (total_bonds, num_bonds), to keep track of all the bond feature vectors,
                belonging to a particular molecule.

        Returns:
             mol_vecs: troch.tensor (shape: batch_size x hidden_size
        """
        # implement edge gate computation
        edge_gate_x = torch.index_select(input=atom_layer_output, dim=0, index=bond_atom_adjacency_graph[:, 0])
        edge_gate_y = torch.index_select(input=atom_layer_output, dim=0, index=bond_atom_adjacency_graph[:, 1])

        edge_gate_synaptic_input = self.U(bond_layer_output) + self.V(edge_gate_x) + self.W(edge_gate_y)

        # apply sigmoid activation for computing edge gates
        edge_gates = F.sigmoid(edge_gate_synaptic_input)

        assert(edge_gates.shape == bond_layer_output.shape)

        # multiply bond features with edge gates
        edge_gates_mul_bond_features_tensor = edge_gates * self.A(bond_layer_output)

        # list to store vector representation for various molecules
        mol_vecs = []
        for start_idx, len in scope:
            mol_vec = edge_gates_mul_bond_features_tensor[start_idx : start_idx + len].sum(dim=0)
            mol_vecs.append(mol_vec)

        # stack all all the mol_vecs into a 2-D tensor
        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs

    # def forward(self, smiles_batch):
    #     """
    #     Description: This method implements the forward pass, for the Molecular Graph Encoder.
    #
    #     Args:
    #         smiles_batch: List[str]
    #             The list of SMILES representations of the molecules, in the batch.
    #
    #     Returns: torch.tensor (shape: batch_size x hidden_size)
    #         The hidden vectors representations of the all the molecular graphs in the batch.
    #     """
    #     # atom_feature_matrix, bond_feature_matrix, atom_adjacency_list, bond_atom_idx_list, scope = self.smiles_batch_to_matrices_and_lists(smiles_batch)
    #
    #     atom_feature_matrix, bond_feature_matrix, atom_adjacency_list, atom_bond_adjacency_list, bond_component_atom_list, scope = \
    #         self.smiles_batch_to_matrices_and_lists(smiles_batch)
    #
    #     atom_feature_matrix = create_var(atom_feature_matrix)
    #     bond_feature_matrix = create_var(bond_feature_matrix)
    #
    #     # obtain feature embedding for all the atom features
    #     # atom_feat_embeding_matrix = self.atom_embedding(atom_feature_matrix)
    #
    #     # obtain feature embedding for all the bond features
    #     # bond_feat_embeding_matrix = self.bond_embedding(bond_feature_matrix)
    #
    #     # implement convolution
    #     atom_layer_input = atom_feature_matrix
    #     bond_layer_input = bond_feature_matrix
    #
    #     for conv_layer in self.conv_layers:
    #         # implement forward pass for this convolutinal layer
    #         atom_layer_output, bond_layer_output = conv_layer(atom_layer_input, bond_layer_input, atom_adjacency_list, atom_bond_adjacency_list)
    #
    #         # set the input features for the next convolutional layer
    #         atom_layer_input, bond_layer_input = atom_layer_output, bond_layer_output
    #
    #     # for each molecular graph, pool all the edge feature vectors
    #     mol_vecs = self.pool_bond_features_for_mols(atom_layer_output, bond_layer_output, bond_component_atom_list, scope)
    #
    #     return mol_vecs

    # def pool_bond_features_for_mols(self, atom_layer_output, bond_layer_output, bond_component_atom_list, scope):
    #     """
    #     Description: For each molecule, pools (akin to max-pooling in convolutional networks) all the edge vectors into a single
    #     representative vector.
    #
    #     Args:
    #         atom_layer_output: torch.tensor (shape num_atoms x hidden_size)
    #             The output atom features from the graph convolutional network.
    #
    #         bond_layer_output: torch.tensor( shape num_bonds x hidden_size)
    #             The output bond features from the graph convolutional network.
    #
    #         bond_component_atom_list: List[torch.tensor]
    #             For each bond, across the entire batch, the idxs of the 2 atoms, of which the bond is composed of.
    #             # Role: For purposes of "edge pooling"
    #
    #     Returns:
    #         mol_vecs: torch.tensor (shape batch_size x hidden_size)
    #             The hidden vectors representations of the all the molecular graphs in the batch.
    #
    #     """
    #     # get total number of bonds in the batch
    #     total_bonds = bond_layer_output.shape[0]
    #
    #     edge_gate_prod_bond_feature_vec = create_var(torch.zeros(total_bonds, self.hidden_size))
    #
    #     for bond_idx in range(total_bonds):
    #         # get the idx of component atoms, of which this bond is composed of
    #         component_atom_idx = bond_component_atom_list[bond_idx]
    #
    #         # retrieve feature vector for atom 'i'
    #         atom_i_feature_vec = atom_layer_output[component_atom_idx[0]]
    #
    #         # retrieve feature vector for atom 'j'
    #         atom_j_feature_vec = atom_layer_output[component_atom_idx[1]]
    #
    #         # retrieve the feature vector for this bond
    #         bond_feature_vec = bond_layer_output[bond_idx]
    #
    #         # compute the edge gate for this bond
    #         edge_gate = self.compute_edge_gate_for_bond(bond_feature_vec, atom_i_feature_vec, atom_j_feature_vec)
    #
    #         # as per Prof. Bresson's Code
    #         edge_gate_prod_bond_feature_vec[bond_idx] = edge_gate * self.A(bond_feature_vec)
    #
    #     # pool all the edges belonging to a molecule
    #     mol_vecs = []
    #     for start, len in scope:
    #         # the molecule vector for molecule, is the mean of the hidden vectors for all the atoms of that
    #         # molecule
    #         mol_vec = edge_gate_prod_bond_feature_vec.narrow(0, start, len).sum(dim=0) / len
    #         mol_vecs.append(mol_vec)
    #
    #     mol_vecs = torch.stack(mol_vecs, dim=0)
    #     return mol_vecs

    # def compute_edge_gate_for_bond(self, bond_feature_vec, atom_i_feature_vec, atom_j_feature_vec):
    #     """
    #     Description: Compute the edge gate for the given bond e_(i, j).
    #
    #     Args:
    #         bond_feature_vec: torch.tensor (shape 1 x hidden_size)
    #             The feature vector for the bond, for whom the edge gate has to be computed.
    #
    #         atom_i_feature_vec: torch.tensor (shape 1 x hidden_size)
    #             The feature vector for atom 'i'.
    #
    #         atom_j_feature_vec: torch.tensor (shape 1 x hidden_size)
    #             The feature vector for atom 'j'.
    #
    #     Returns:
    #         edge_gate: torch.tensor (shape 1 x hidden_size)
    #             The edge gate for the corresponding bond / edge e_(i, j).
    #     """
    #     # as per Prof. Bresson's code
    #     Ue = self.U(bond_feature_vec)
    #
    #     Vx = self.V(atom_i_feature_vec)
    #
    #     Wx = self.W(atom_j_feature_vec)
    #
    #     edge_gate = Ue + Vx + Wx
    #
    #     # apply sigmoid activation
    #     edge_gate = nn.Sigmoid()(edge_gate)
    #
    #     return edge_gate

    @staticmethod
    def tensorize(smiles_batch):
        """
        Args:
            smiles_batch: List[str]
                The list of SMILES representations of the molecules, in the batch.

        Returns:
            atom_feature_matrix: torch.tensor (shape: batch_size x atom_feature_dim)
                The matrix containing feature vectors, for all the atoms, across the entire batch.
                * atom_feature_dim = len(ELEM_LIST) + 6 + 5 + 4 + 1

            bond_feature_matrix: torch.tensor (shape: batch_size x bond_feature_dim)
                The matrix containing feature vectors, for all the bonds, across the entire batch.
                * bond_feature_dim = 5 + 6

            atom_adjacency_graph: torch.tensor (shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For each atom, across the entire batch, the idxs of neighboring atoms.

            atom_bond_adjacency_graph: torch.tensor(shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For each atom, across the entire batch, the idxs of all the bonds, in which it is the initial atom.

            bond_component_atom_graph: torch.tensor (shape num_bonds x 2)
                For each bond, across the entire batch, the idxs of the 2 atoms, of which the bond is composed of.

            scope: List[Tuple(int, int)]
                The list to store tuples (total_bonds, num_bonds), to keep track of all the bond feature vectors,
                belonging to a particular molecule.
        """
        # list to store feature vectors, for all the atoms, across the entire batch
        atom_feature_vecs = []

        # padding vector for atom features
        atom_feature_padding = torch.zeros(ATOM_FEATURE_DIM)

        # append padding vector to list
        atom_feature_vecs.append(atom_feature_padding)

        # we refer to each atom, across the entire batch, with an idx
        # atom idxs are 1-indexed, because first feature vector is the padding vector
        atom_idx_offset = 1

        # list to store, the list of idxs of neighboring atoms, for all the atoms, across the entire batch
        atom_adjacency_list = []

        atom_adjacency_list.append([0] * MAX_NUM_NEIGHBORS)

        # list to store, for each atom, across the entire batch, the list of idxs of all the bonds, in which it is the initial atom
        atom_bond_adjacency_list = []

        atom_bond_adjacency_list.append([0] * MAX_NUM_NEIGHBORS)

        # list to store, feature vectors, for all the bonds, across the entire batch
        bond_feature_vecs = []

        # padding vector for bond features
        bond_feature_padding = torch.zeros(BOND_FEATURE_DIM)

        # append padding vector to list
        bond_feature_vecs.append(bond_feature_padding)

        # for each bond, across the entire batch, the idxs of the 2 atoms, of which the bond is composed of.
        bond_atom_adjacency_list = []

        bond_atom_adjacency_list.append([0] * 2)

        # we index all bonds, across the entire batch
        # bond idxs are 1-indexed, because first feature vector is the padding vector
        bond_idx_offset = 1
        
        # list to store tuples (start_idx, len), to keep track of bond feature vectors,
        # belonging to a particular molecule
        scope = []

        # set the start_idx for keeping track of all the bonds belonging to a molecule
        # bond idxs are 1-indexed, because first feature vector is the padding vector
        bond_start_idx = 1

        for smiles in smiles_batch:
            # get the corresponding molecule from the SMILES representation
            mol = get_kekulized_mol_from_smiles(smiles)

            for atom in mol.GetAtoms():
                # get the feature vector for this atom
                atom_feature_vec = get_atom_feature_vec(atom)

                # append this feature vector to the list
                atom_feature_vecs.append(atom_feature_vec)

                # append a sublist to store idxs of neighboring atoms for this atom
                atom_adjacency_list.append([])

                # append a sublist to store idxs of all bonds, in which it is the initial atom
                atom_bond_adjacency_list.append([])

            for bond in mol.GetBonds():
                bond_begin_atom = bond.GetBeginAtom()
                bond_end_atom = bond.GetEndAtom()

                # get the offsetted idxs, of these atoms, across the batch
                offset_begin_idx = bond_begin_atom.GetIdx() + atom_idx_offset
                offset_end_idx = bond_end_atom.GetIdx() + atom_idx_offset

                # append idxs of neighboring atoms, to appropriate lists
                atom_adjacency_list[offset_begin_idx].append(offset_end_idx)
                atom_adjacency_list[offset_end_idx].append(offset_begin_idx)

                # get the feature vector for this bond
                bond_feature_vec = get_bond_feature_vec(bond)

                # for edge e_(i, j)
                component_atom_idx_list = [offset_begin_idx, offset_end_idx]
                bond_atom_adjacency_list.append(component_atom_idx_list)

                # append the bond idx to the atom_bond_adjacency_list sublist for offset_begin_idx
                atom_bond_adjacency_list[offset_begin_idx].append(bond_idx_offset)

                # append bond feature vector to the list
                bond_feature_vecs.append(bond_feature_vec)

                # increment idx
                bond_idx_offset += 1

                # for edge e_(j, i)
                component_atom_idx_list = [offset_end_idx, offset_begin_idx]
                bond_atom_adjacency_list.append(component_atom_idx_list)

                # append the bond idx to the atom_bond_adjacency_list sublist for offset_end_idx
                atom_bond_adjacency_list[offset_end_idx].append(bond_idx_offset)

                # append bond feature vector to the list
                bond_feature_vecs.append(bond_feature_vec)

                # increment idx
                bond_idx_offset += 1

            # append scope for this molecule
            # each bond / edge is counted twice i.e. e_(i, j) and e_(j, i)
            _len = 2 * mol.GetNumBonds()

            scope.append((bond_start_idx, _len))

            bond_start_idx += _len

            # update atom_idx_offset
            atom_idx_offset += mol.GetNumAtoms()
            
        # obtain the total number of atoms across the batch
        total_num_atoms = len(atom_adjacency_list)

        # obtain the total number of bonds across the batch
        total_num_bonds = len(bond_atom_adjacency_list)

        # obtain the matrix of all atom features
        atom_feature_matrix = torch.stack(atom_feature_vecs, dim=0)

        # obtain the matrix of all bond features
        bond_feature_matrix = torch.stack(bond_feature_vecs, dim=0)
        
        # initialize the adjacency graph for all the atoms, across the entire batch
        atom_adjacency_graph = torch.zeros(total_num_atoms, MAX_NUM_NEIGHBORS).long()

        # initialize the adjacency graph for all the atoms, across the entire batch
        atom_bond_adjacency_graph = torch.zeros(total_num_atoms, MAX_NUM_NEIGHBORS).long()

        # initialize the adjacency graph for all the bonds, across the entire batch
        bond_atom_adjacency_graph = torch.zeros(total_num_bonds, 2).long()

        # insert values into the atom adjacency graph
        for atom_idx in range(total_num_atoms):
            for idx, neighbor_atom_idx in enumerate(atom_adjacency_list[atom_idx]):
                atom_adjacency_graph[atom_idx, idx] = neighbor_atom_idx

        # insert values into the atom bond adjacency graph
        for atom_idx in range(total_num_atoms):
            for idx, bond_idx in enumerate(atom_bond_adjacency_list[atom_idx]):
                atom_bond_adjacency_graph[atom_idx, idx] = bond_idx

        # insert values into the bond atom adjacency graph
        for bond_idx in range(total_num_bonds):
            for idx, atom_idx in enumerate(bond_atom_adjacency_list[bond_idx]):
                bond_atom_adjacency_graph[bond_idx, idx] = atom_idx

        # return atom_feature_matrix, bond_feature_matrix, atom_adjacency_list, bond_atom_idx_list, scope
        # return atom_feature_matrix, bond_feature_matrix, atom_adjacency_list, atom_bond_adjacency_list, bond_component_atom_list, scope
        return (atom_feature_matrix, bond_feature_matrix, atom_adjacency_graph, atom_bond_adjacency_graph, bond_atom_adjacency_graph, scope)

