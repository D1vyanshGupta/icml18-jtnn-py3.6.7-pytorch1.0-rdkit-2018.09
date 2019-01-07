import torch
import torch.nn as nn

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

class MolGraphEncoder(nn.Module):
    """
    Description: This class implements a "Graph Convolutional Neural Network" for encoding molecular graphs.
    """

    def __init__(self, hidden_size, num_layers, atom_embedding=None, bond_embedding=None):
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

        # setup embedding space for encoding the the high-dimensional and sparse atom feature vectors,
        # into low-dimensional and dense feature vectors

        atom_feat_space_size = len(ELEM_LIST) * 6 * 5 * 4 * 1

        if atom_embedding is None:
            self.atom_embedding = nn.Embedding(atom_feat_space_size, hidden_size)
        else:
            self.atom_embedding = atom_embedding

        # setup embedding space for encoding the the high-dimensional and sparse bond feature vectors,
        # into low-dimensional and dense feature vectors

        bond_feat_space_size = 5 * 6

        if bond_embedding is None:
            self.bond_embedding = nn.Embedding(bond_feat_space_size, hidden_size)
        else:
            self.bond_embedding = bond_embedding

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

    def forward(self, smiles_batch):
        """
        Description: This method implements the forward pass, for the Molecular Graph Encoder.

        Args:
            smiles_batch: List[str]
                The list of SMILES representations of the molecules, in the batch.

        Returns: torch.tensor (shape: batch_size x hidden_size)
            The hidden vectors representations of the all the molecular graphs in the batch.
        """
        # atom_feature_matrix, bond_feature_matrix, atom_adjacency_list, bond_atom_idx_list, scope = self.smiles_batch_to_matrices_and_lists(smiles_batch)

        atom_feature_matrix, bond_feature_matrix, atom_adjacency_list, atom_bond_adjacency_list, bond_component_atom_list, scope = \
            self.smiles_batch_to_matrices_and_lists(smiles_batch)

        atom_feature_matrix = create_var(atom_feature_matrix)
        bond_feature_matrix = create_var(bond_feature_matrix)

        # obtain feature embedding for all the atom features
        # atom_feat_embeding_matrix = self.atom_embedding(atom_feature_matrix)

        # obtain feature embedding for all the bond features
        # bond_feat_embeding_matrix = self.bond_embedding(bond_feature_matrix)

        # implement convolution
        atom_layer_input = atom_feature_matrix
        bond_layer_input = bond_feature_matrix

        for conv_layer in self.conv_layers:
            # implement forward pass for this convolutinal layer
            atom_layer_output, bond_layer_output = conv_layer(atom_layer_input, bond_layer_input, atom_adjacency_list, atom_bond_adjacency_list)

            # set the input features for the next convolutional layer
            atom_layer_input, bond_layer_input = atom_layer_output, bond_layer_output

        # for each molecular graph, pool all the edge feature vectors
        mol_vecs = self.pool_bond_features_for_mols(atom_layer_output, bond_layer_output, bond_component_atom_list, scope)

        return mol_vecs

    def pool_bond_features_for_mols(self, atom_layer_output, bond_layer_output, bond_component_atom_list, scope):
        """
        Description: For each molecule, pools (akin to max-pooling in convolutional networks) all the edge vectors into a single
        representative vector.

        Args:
            atom_layer_output: torch.tensor (shape num_atoms x hidden_size)
                The output atom features from the graph convolutional network.

            bond_layer_output: torch.tensor( shape num_bonds x hidden_size)
                The output bond features from the graph convolutional network.

            bond_component_atom_list: List[torch.tensor]
                For each bond, across the entire dataset, the idxs of the 2 atoms, of which the bond is composed of.
                # Role: For purposes of "edge pooling"

        Returns:
            mol_vecs: torch.tensor (shape batch_size x hidden_size)
                The hidden vectors representations of the all the molecular graphs in the batch.

        """
        # get total number of bonds in the batch
        total_bonds = bond_layer_output.shape[0]

        edge_gate_prod_bond_feature_vec = create_var(torch.zeros(total_bonds, self.hidden_size))

        for bond_idx in range(total_bonds):
            # get the idx of component atoms, of which this bond is composed of
            component_atom_idx = bond_component_atom_list[bond_idx]

            # retrieve feature vector for atom 'i'
            atom_i_feature_vec = atom_layer_output[component_atom_idx[0]]

            # retrieve feature vector for atom 'j'
            atom_j_feature_vec = atom_layer_output[component_atom_idx[1]]

            # retrieve the feature vector for this bond
            bond_feature_vec = bond_layer_output[bond_idx]

            # compute the edge gate for this bond
            edge_gate = self.compute_edge_gate_for_bond(bond_feature_vec, atom_i_feature_vec, atom_j_feature_vec)

            # as per Prof. Bresson's Code
            edge_gate_prod_bond_feature_vec[bond_idx] = edge_gate * self.A(bond_feature_vec)

        # pool all the edges belonging to a molecule
        mol_vecs = []
        for start, len in scope:
            # the molecule vector for molecule, is the mean of the hidden vectors for all the atoms of that
            # molecule
            mol_vec = edge_gate_prod_bond_feature_vec.narrow(0, start, len).sum(dim=0) / len
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

    def compute_edge_gate_for_bond(self, bond_feature_vec, atom_i_feature_vec, atom_j_feature_vec):
        """
        Description: Compute the edge gate for the given bond e_(i, j).

        Args:
            bond_feature_vec: torch.tensor (shape 1 x hidden_size)
                The feature vector for the bond, for whom the edge gate has to be computed.

            atom_i_feature_vec: torch.tensor (shape 1 x hidden_size)
                The feature vector for atom 'i'.

            atom_j_feature_vec: torch.tensor (shape 1 x hidden_size)
                The feature vector for atom 'j'.

        Returns:
            edge_gate: torch.tensor (shape 1 x hidden_size)
                The edge gate for the corresponding bond / edge e_(i, j).
        """
        # as per Prof. Bresson's code
        Ue = self.U(bond_feature_vec)

        Vx = self.V(atom_i_feature_vec)

        Wx = self.W(atom_j_feature_vec)

        edge_gate = Ue + Vx + Wx

        # apply sigmoid activation
        edge_gate = nn.Sigmoid()(edge_gate)

        return edge_gate

    def smiles_batch_to_matrices_and_lists(self, smiles_batch):
        """
        Description: This method, given the batch of SMILES representations, of the molecules,
        returns the various matrices / lists encoding relevant information, as described below.

        Args:
            smiles_batch: List[str]
                The list of SMILES representations of the molecules, in the batch.

        Returns:

            atom_feature_matrix: torch.tensor (shape: batch_size x atom_feature_dim)
                The matrix containing feature vectors, for all the atoms, across the entire dataset.
                * atom_feature_dim = len(ELEM_LIST) + 6 + 5 + 4 + 1

            bond_feature_matrix: torch.tensor (shape: batch_size x bond_feature_dim)
                The matrix containing feature vectors, for all the bonds, across the entire dataset.
                * bond_feature_dim = 5 + 6

            atom_adjacency_list: List[torch.tensor]
                The adjacency list, for all the atoms, across the entire dataset.
                Each adjacency list contains the idxs of neighboring atoms.

            atom_bond_adjacency_list: List[torch.tensor]
                For each atom, across the entire dataset, the idxs of all the bonds, in which this atom is present.
                * Role: For purposes of "edge gate" computation.

            bond_component_atom_list: List[torch.tensor]
                For each bond, across the entire dataset, the idxs of the 2 atoms, of which the bond is composed of.
                # Role: For purposes of "edge pooling"

            scope: List[Tuple(int, int)]
                The list to store tuples (total_bonds, num_bonds), to keep track of all the bond feature vectors,
                belonging to a particular molecule.
        """
        # we refer to each atom, across the entire dataset, with an idx
        atom_idx_offset = 0

        # we refer to each bond (bonds are undirected edges), across the entire dataset, with an idx
        bond_idx_offset = 0

        # list to store feature vectors, for all the atoms, across the entire dataset
        atom_feature_vecs = []

        # list to store, the list of idxs of neighboring atoms, for all the atoms, across the entire dataset
        atom_adjacency_list = []

        # list to store, the list of idxs of all bonds, in which this atom is present,
        # for all the atoms, across the entire dataset
        atom_bond_adjacency_list = []

        # For each bond, across the entire dataset, the idxs of the 2 atoms, of which the bond is composed of.
        bond_component_atom_list = []

        # list to store, feature vectors, for all the bonds, across the entire dataset
        bond_feature_vecs = []

        # list to store tuples (total_atoms, num_atoms), to keep track of feature vectors,
        # belonging to a particular molecule
        scope = []

        # set the start for keeping track of all the bonds belonging to a molecule
        start = 0

        for smiles in smiles_batch:
            # get the corresponding molecule from the SMILES representation
            mol = get_kekulized_mol_from_smiles(smiles)


            for atom in mol.GetAtoms():
                # get the feature vector for this atom
                atom_feature_vec = self.get_atom_feature_vec(atom)

                # append this feature vector to the list
                atom_feature_vecs.append(atom_feature_vec)

                atom_adjacency_list.append([])

                atom_bond_adjacency_list.append([])

            for bond in mol.GetBonds():
                bond_begin_atom = bond.GetBeginAtom()
                bond_end_atom = bond.GetEndAtom()

                # get the offsetted idxs, of these atoms, across the dataset
                offset_begin_idx = bond_begin_atom.GetIdx() + atom_idx_offset
                offset_end_idx = bond_end_atom.GetIdx() + atom_idx_offset

                # append idxs of neighboring atoms, to appropriate lists
                atom_adjacency_list[offset_begin_idx].append(offset_end_idx)
                atom_adjacency_list[offset_end_idx].append(offset_begin_idx)

                # get the feature vector for this bond
                bond_feature_vec = self.get_bond_feature_vec(bond)

                # append the idx of bond, in which beginning atom is present
                atom_bond_adjacency_list[offset_begin_idx].append(bond_idx_offset)

                # append this feature vector to the list
                bond_feature_vecs.append(bond_feature_vec)

                # for edge e_(i, j)
                component_atom_idx_list = create_var(torch.tensor([offset_begin_idx, offset_end_idx]))

                bond_component_atom_list.append(component_atom_idx_list)

                bond_idx_offset += 1

                # append the idx of bond, in which ending atom is present
                atom_bond_adjacency_list[offset_end_idx].append(bond_idx_offset)

                # append this feature vector to the list
                bond_feature_vecs.append(bond_feature_vec)

                # for edge e_(j, i)

                component_atom_idx_list = create_var(torch.LongTensor([offset_end_idx, offset_begin_idx]))

                bond_component_atom_list.append(component_atom_idx_list)

                bond_idx_offset += 1

            # append scope for this molecule
            # each bond / edge is counted twice i.e. e_(i, j) and e_(j, i)
            len = 2 * mol.GetNumBonds()

            scope.append((start, len))

            start += len

            # update atom
            atom_idx_offset += mol.GetNumAtoms()

        # obtain the matrix of all atom features
        atom_feature_matrix = torch.stack(atom_feature_vecs, dim=0)

        # obtain the matrix of all bond features
        bond_feature_matrix = torch.stack(bond_feature_vecs, dim=0)

        # convert the list of idxs to tensors, for torch.index_select operation
        atom_adjacency_list = list(map(lambda x: create_var(torch.LongTensor(x)), atom_adjacency_list))

        # convert the list of idxs to tensors, for torch.index_select operation
        atom_bond_adjacency_list = list(map(lambda x: create_var(torch.LongTensor(x)), atom_bond_adjacency_list))

        # return atom_feature_matrix, bond_feature_matrix, atom_adjacency_list, bond_atom_idx_list, scope
        return atom_feature_matrix, bond_feature_matrix, atom_adjacency_list, atom_bond_adjacency_list, bond_component_atom_list, scope
    
    def one_hot_encode(self, x, allowable_set):
        """
        Description: This method, given a categorical variable,
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

    def get_atom_feature_vec(self, atom):
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
            self.one_hot_encode(atom.GetSymbol(), ELEM_LIST)
            # one-hot encode atom degree
            + self.one_hot_encode(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            # one-hot encode formal charge of the atom
            + self.one_hot_encode(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
            # one-hot encode the chiral tag of the atom
            + self.one_hot_encode(int(atom.GetChiralTag()), [0, 1, 2, 3])
            # one-hot encoding / binary encoding whether atom is aromatic or not
            + [atom.GetIsAromatic()]
        )

    def get_bond_feature_vec(self, bond):
        """
        Description: This method, constructs the feature vector for the given bond.

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
        stereo_conf_feature = self.one_hot_encode(stereo, [0, 1, 2, 3, 4, 5])
        return torch.Tensor(bond_type_feature + stereo_conf_feature)
