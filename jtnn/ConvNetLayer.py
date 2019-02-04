import torch
import torch.nn as nn
import torch.nn.functional as F

from nnutils import create_var, index_select_ND

class ConvNetLayer(nn.Module):
    """
    Description: This class implements a single "Graph Convolutional Layer" in the "Graph ConvNet Encoder" architecture.
    """

    def __init__(self, hidden_size, atom_feature_dim=0, bond_feature_dim=0):
        """
        Description: Constructor for the class.

        Args:
            hidden_size: int
                The dimension of the hidden feature vectors to be used.

            atom_feature_dim: int
                The dimension of the atom feature vectors.

            bond_feature_dim: int
                The dimension of the bond feature vectors.
        """
        # invoke the superclass constructor
        super(ConvNetLayer, self).__init__()

        # the dimension of the hidden vectors
        self.hidden_size = hidden_size

        self.atom_feature_dim = atom_feature_dim

        self.bond_feature_dim = bond_feature_dim

        # batch normalization for bond/edge features
        self.bn_bond_features = nn.BatchNorm1d(num_features=hidden_size)

        # batch normalization for atom/node features
        self.bn_atom_features = nn.BatchNorm1d(num_features=hidden_size)


        if atom_feature_dim != 0 and bond_feature_dim != 0:
            # weight matrices for the node features (as per Prof. Bresson's equations)
            self.U = nn.Linear(atom_feature_dim, hidden_size, bias=True)

            self.V = nn.Linear(atom_feature_dim, hidden_size, bias=True)

            # weight matrices for "edge gate" computation (as per Prof. Bresson's code)
            self.A = nn.Linear(bond_feature_dim, hidden_size, bias=True)

            self.B = nn.Linear(atom_feature_dim, hidden_size, bias=True)

            self.C = nn.Linear(atom_feature_dim, hidden_size, bias=True)

        else:
            # weight matrices for the node features (as per Prof. Bresson's equations)
            self.U = nn.Linear(hidden_size, hidden_size, bias=True)

            self.V = nn.Linear(hidden_size, hidden_size, bias=True)

            # weight matrices for "edge gate" computation (as per Prof. Bresson's code)
            self.A = nn.Linear(hidden_size, hidden_size, bias=True)

            self.B = nn.Linear(hidden_size, hidden_size, bias=True)

            self.C = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, atom_layer_input, bond_layer_input, atom_adjacency_graph, atom_bond_adjacency_graph, bond_atom_adjacency_graph):
        """
        Args:
            atom_layer_input: torch.tensor (shape: batch_size x atom_feature_dim)
                The matrix containing feature vectors, for all the atoms, across the entire batch.
                * atom_feature_dim = len(ELEM_LIST) + 6 + 5 + 4 + 1

            bond_layer_input: torch.tensor (shape: batch_size x bond_feature_dim)
                The matrix containing feature vectors, for all the bonds, across the entire batch.
                * bond_feature_dim = 5 + 6

            atom_adjacency_graph: torch.tensor (shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For each atom, across the entire batch, the idxs of neighboring atoms.

            atom_bond_adjacency_graph: torch.tensor(shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For each atom, across the entire batch, the idxs of all the bonds, in which it is the initial atom.

            bond_atom_adjacency_graph: torch.tensor (shape num_bonds x 2)
                For each bond, across the entire batch, the idxs of the 2 atoms, of which the bond is composed of.
        """
        # implement edge gate computation
        edge_gate_x = torch.index_select(input=atom_layer_input, dim=0, index=bond_atom_adjacency_graph[:, 0])
        edge_gate_y = torch.index_select(input=atom_layer_input, dim=0, index=bond_atom_adjacency_graph[:, 1])

        assert(bond_layer_input.shape[0] == edge_gate_x.shape[0])
        assert (bond_layer_input.shape[0] == edge_gate_y.shape[0])

        edge_gate_synaptic_input = self.A(bond_layer_input) + self.B(edge_gate_x) + self.C(edge_gate_y)

        # apply sigmoid activation for computing edge gates
        edge_gates = F.sigmoid(edge_gate_synaptic_input)

        # implement batch normalization for bond/edge features
        edge_gate_synaptic_input = self.bn_bond_features(edge_gate_synaptic_input)

        # apply ReLU activation for computing new bond features
        # add residual
        if self.bond_feature_dim == 0:
            bond_layer_output = F.relu(edge_gate_synaptic_input) + bond_layer_input
        else:
            bond_layer_output = F.relu(edge_gate_synaptic_input)

        bond_layer_output = F.relu(edge_gate_synaptic_input)

        # implement node features computation

        # for each atom, aggregate the features vectors of neighboring atoms
        atom_neighbor_features_tensor = self.V(index_select_ND(atom_layer_input, 0, atom_adjacency_graph))

        # for each atom, get the edge gates for the corresponding neighbor atom features
        atom_neighbor_edge_gates_tensor = index_select_ND(edge_gates, 0, atom_bond_adjacency_graph)

        assert(atom_neighbor_edge_gates_tensor.shape == atom_neighbor_features_tensor.shape)

        # for each atom, multiply the edge gates with corresponding neighbor atom feature vectors
        atom_neighbor_message_tensor = atom_neighbor_edge_gates_tensor * atom_neighbor_features_tensor

        atom_neighbor_message_sum = atom_neighbor_message_tensor.sum(dim=1)

        assert(atom_neighbor_message_sum.shape[0] == atom_layer_input.shape[0])

        atom_features_synaptic_input = self.U(atom_layer_input) + atom_neighbor_message_sum

        # implement batch normalization
        atom_features_synaptic_input = self.bn_atom_features(atom_features_synaptic_input)

        # apply ReLU activation for computing new atom features
        # add residual
        if self.atom_feature_dim == 0:
            atom_layer_output = F.relu(atom_features_synaptic_input) + atom_layer_input
        else:
            atom_layer_output = F.relu(atom_features_synaptic_input)

        # atom_layer_output = F.relu(atom_features_synaptic_input)

        return atom_layer_output, bond_layer_output


        # for atom_idx in range(total_atoms):
        #     # feature vector for the current atom
        #     atom_feature_vec = atom_feature_matrix[atom_idx]
        #
        #     # idxs of all the neighbor atoms, of current atom
        #     neighbor_atom_idx = atom_adjacency_list[atom_idx]
        #
        #     # feature vectors of all the neighbor atoms.
        #     neighbor_atom_feature_vecs = torch.index_select(input=atom_feature_matrix, dim=0, index=neighbor_atom_idx)
        #
        #     # idxs of all the bonds, in which this atom is that beginning atom
        #     bond_atom_idx = atom_bond_adjacency_list[atom_idx]
        #
        #     # feature vectors of all the bonds, in which this atom is that beginning atom
        #     bond_feature_vecs = torch.index_select(input=bond_feature_matrix, dim=0, index=bond_atom_idx)
        #
        #     # compute new bond features, of all the bond, in which this atom, is the starting atom
        #     bond_features_synaptic_input = self.evaluate_bond_features_synaptic_input(atom_feature_vec, bond_feature_vecs, neighbor_atom_feature_vecs)
        #
        #     # apply the ReLU activation onto the new edge
        #     new_bond_features = nn.ReLU()(bond_features_synaptic_input)
        #
        #     # update the feature vectors of all the bonds, in which this atom is the beginning atom
        #     bond_layer_output[bond_atom_idx] = new_bond_features
        #
        #     # evaluate the edge gates, for all the edges in which this atom is the beginning atom
        #     edge_gates = nn.Sigmoid()(bond_features_synaptic_input)
        #
        #     # implement point-wise multiplication (Hadamard Product)
        #     edge_gate_prod_neighbor_vecs = edge_gates * self.V(neighbor_atom_feature_vecs)
        #
        #     # sum up the hadamard product of the edge gates and the neighbor atom feature vectors
        #     neighbor_vec_edge_gate_sum = torch.sum(edge_gate_prod_neighbor_vecs, dim=0)
        #
        #     # evaluate the new feature vector for the atom
        #     new_atom_vec_synaptic_input = self.U(atom_feature_vec) + neighbor_vec_edge_gate_sum
        #
        #     # apply ReLU activation
        #     new_atom_vec = nn.ReLU()(new_atom_vec_synaptic_input)
        #
        #     # set the atom's feature vector to the new value
        #     atom_layer_output[atom_idx] = new_atom_vec
        return atom_layer_output, bond_layer_output

    # def evaluate_bond_features_synaptic_input(self, atom_feature_vec, bond_feature_vecs, neighbor_atom_feature_vecs):
    #     """
    #     Description: This method, in reference to atom with idx 'i', computes the new values (synaptic input)
    #     for all the edges of the form e_(i,j), where j refers to idxs of all the atom, that are neighbors of
    #     atom with idx i.
    #
    #     Args:
    #         atom_feature_vec: torch.tensor (shape: atom_feature_dim)
    #             The feature vector of atom with idx 'i'.
    #
    #         bond_feature_vecs: torch.tensor (shape: num_neighbors x bond_feature_dim)
    #             The feature vectors of all the bonds of the form e_(i,j).
    #
    #         neighbor_atom_feature_vecs: torch.tensor (shape: num_neighbors x atom_feature_dim)
    #             The feature vectors of all the atoms with idxs 'j' i.e. idxs of neighboring atoms, of atom with idx 'i'.
    #     """
    #
    #     # multiply the edge feature vectors with the A matrix
    #     Ae = self.A(bond_feature_vecs)
    #
    #     # number of neighbors of atom with idx 'i'
    #     num_neighbors = bond_feature_vecs.shape[0]
    #
    #     repeated_atom_feature_vecs = [atom_feature_vec for idx in range(num_neighbors)]
    #     repeated_atom_feature_vecs = torch.stack(repeated_atom_feature_vecs , dim=0)
    #
    #     # multiply atom feature vectors with the B matrix
    #     Bx = self.B(repeated_atom_feature_vecs)
    #
    #     # multiply the neighboring atom feature vectors with the C matrix
    #     Cx = self.C(neighbor_atom_feature_vecs)
    #
    #     # as per Prof. Bresson's code
    #     bond_features_synaptic_input = Ae + Bx + Cx
    #
    #     return bond_features_synaptic_input

