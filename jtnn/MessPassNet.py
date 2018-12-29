import rdkit.Chem as Chem
from nnutils import *
from chemutils import get_kekulized_mol_from_smiles

# list of elements are were are considering in our problem domain
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

# dimension of the atom feature vector
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1

# one-hot encoding for the atom / element type
# one-hot encoding for the degree of the atom ([0, 1, 2, 3, 4, 5])
# one-hot encoding for the formal charge of the atom ([-2, -1, 0, 1, 2])
# one-hot enoding for the chiral-tag of the atom i.e. number of chiral centres ([0, 1, 2, 3])
# one-hot encoding / binary encoding whether atom is aromatic or not
ATOM_FEATURE_DIM = len(ELEM_LIST) + 6 + 5 + 4 + 1

# dimension of the bond feature vector
BOND_FDIM = 5 + 6

# one-hot encoding for the bond-type ([single-bond, double-bond, triple-bond, aromatic-bond, in-ring])
# one-hot encoding for the stereo-configuration of the bond ([0, 1, 2, 3, 4, 5])
BOND_FEATURE_DIM = 5 + 6

# maximum number of neighbors of "cluster-node" in "cluster-graph"
MAX_NUM_NEIGHBORS = 6

# message passing network
class MessPassNet(nn.Module):
    """
    Message Passing Network for encoding molecular graphs.
    """
    # class constructor
    def __init__(self, hidden_size, depth):
        """
        Constructor for the class.

        Args:
            hidden_size: Dimension of the encoding space.
            depth: Number of timesteps for which to run the message passing

        Returns:
            The corresponding MessPassNet object.
        """

        # invoke superclass constructor
        super(MessPassNet, self).__init__()

        # size of hidden "edge message vectors"
        self.hidden_size = hidden_size

        # number of timesteps for which to run the message passing
        self.depth = depth

        # weight matrix for bond feature matrix
        # instead of W^g_1 x_u + W^g_2 x_uv
        # concatenate x_u and x_uv and use W_i x
        self.W_i = nn.Linear(ATOM_FEATURE_DIM + BOND_FEATURE_DIM, hidden_size, bias=False)
        # weight matrix for hidden layer
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        # weight matrix for output layer
        # instead of U^g_1 x_u + summation U^g_2 nu^(T)_vu
        # concatenate x_u and summation nu^(T)_vu and use W_o x
        self.W_o = nn.Linear(ATOM_FEATURE_DIM + hidden_size, hidden_size)

    def forward(self, smiles_batch):
        """
        Args:
            smiles_batch: list / batch of SMILES representations in the entire training dataset.

        Returns:
             mol_vecs: Encoding vectors for all the molecular graphs in the entire training dataset.
        """
        atom_feature_matrix, bond_feature_matrix, atom_graph, bond_graph, scope = self.smiles_batch_to_feat_matrices(smiles_batch)

        # create pytorch Variables
        atom_feature_matrix = create_var(atom_feature_matrix)
        bond_feature_matrix = create_var(bond_feature_matrix)
        atom_graph = create_var(atom_graph)
        bond_graph = create_var(bond_graph)

        bond_features_synaptic_input = self.W_i(bond_feature_matrix)

        # apply ReLU activation for timestep, t = 0
        message = nn.ReLU()(bond_features_synaptic_input)

        # implement message passing for timesteps, t = 1 to T (depth)
        for timestep in range(self.depth - 1):

            # obtain messages from all the neighbor nodes
            neighbor_message_vecs = index_select_ND(message, 0, bond_graph)

            # sum up all the neighbor node message vectors
            neighbor_message_vec_sum = neighbor_message_vecs.sum(dim=1)

            # multiply with the weight matrix for the hidden layer
            neighbor_message = self.W_h(neighbor_message_vec_sum)

            # message at timestep t + 1
            message = nn.ReLU()(bond_features_synaptic_input + neighbor_message)

        # neighbor message vectors for each node from the message matrix
        neighbor_message_vecs = index_select_ND(message, 0, atom_graph)

        # neighbor message for each atom
        neighbor_message_atom_matrix = neighbor_message_vecs.sum(dim=1)

        # concatenate atom feature vector and message vector
        atom_input_matrix = torch.cat([atom_feature_matrix, neighbor_message_atom_matrix], dim=1)
        atom_hidden_layer_activation_matrix = nn.ReLU()(self.W_o(atom_input_matrix))

        # list to store the corresponding molecule vectors for each molecule
        mol_vecs = []
        for st, le in scope:
            # the molecule vector for molecule, is the mean of the hidden vectors for all the atoms of that
            # molecule
            mol_vec = atom_hidden_layer_activation_matrix.narrow(0, st, le).sum(dim=0) / le
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
            # one-hot encode the chiral tag of the atom
            + self.one_hot_encode(int(atom.GetChiralTag()), [0, 1, 2, 3])
            # one-hot encoding / binary encoding whether atom is aromatic or not
            + [atom.GetIsAromatic()]
        )

    def get_bond_feature_vec(self, bond):
        """
        This function, constructs the feature vector for the given bond.
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

    def smiles_batch_to_feat_matrices(self, smiles_batch):
        """
        This method, given a batch of SMILES representations,
        constructs the feature vectors for the corresponding molecules

        Args:
            smiles_batch: The batch of SMILES representations over the training set.

        Returns:
            atom_feature_matrix: Matrix of atom features for all the atoms, over all the molecules, in the dataset.
            bond_feature_matrix: Matrix of bond features for all the bond, over all the molecules, in the dataset.
            atom_graph: For every atom across the training dataset, this atom_graph gives the bond idxs of all the bonds
                        in which it is present. An atom can at most be present in MAX_NUM_NEIGHBORS(= 6) bonds.
            bond_graph: For every non-ring bond (cluster-node) across the training dataset, this bond_graph gives the bond idx of
                        those non-ring bonds (cluster-nodes), to which it is connected in the "cluster-graph".
            scope: List of tuples of (total_atoms, num_atoms). Used to extract the atom / bond features for a
                   particular molecule in the training dataset, from the atom_feature_matrix / bond_feature_matrix.
        """
        # add zero-padding for the bond feature vector matrix
        padding = torch.zeros(ATOM_FEATURE_DIM + BOND_FEATURE_DIM)

        # ensure that the bond feature vectors are 1-indexed
        atom_feature_vecs, bond_feature_vecs = [], [padding]

        # ensure that the bonds are is 1-indexed

        # in_bonds, for a given atom_idx in a molecule, stores the list of idxs of all bonds in which it is present
        in_bonds, all_bonds = [], [(-1, -1)]
        scope = []

        total_atoms = 0

        for smiles in smiles_batch:
            # get the corresponding molecule from the SMILES representation
            mol = get_kekulized_mol_from_smiles(smiles)

            # get the number of atoms in the molecule
            num_atoms = mol.GetNumAtoms()

            for atom in mol.GetAtoms():
                atom_feature_vecs.append(self.get_atom_feature_vec(atom))
                in_bonds.append([])

            for bond in mol.GetBonds():
                bond_begin_atom = bond.GetBeginAtom()
                bond_end_atom = bond.GetEndAtom()

                x = bond_begin_atom.GetIdx() + total_atoms
                y = bond_end_atom.GetIdx() + total_atoms

                num_bonds = len(all_bonds)
                all_bonds.append((x, y))
                bond_feature_vecs.append(torch.cat([atom_feature_vecs[x], self.get_bond_feature_vec(bond)], 0))
                in_bonds[y].append(num_bonds)

                num_bonds = len(all_bonds)
                all_bonds.append((y, x))
                bond_feature_vecs.append(torch.cat([atom_feature_vecs[y], self.get_bond_feature_vec(bond)], 0))
                in_bonds[x].append(num_bonds)

            scope.append((total_atoms, num_atoms))
            total_atoms += num_atoms

        num_bonds = len(all_bonds)
        atom_feature_matrix = torch.stack(atom_feature_vecs, 0)
        bond_feature_matrix = torch.stack(bond_feature_vecs, 0)
        atom_graph = torch.zeros(total_atoms, MAX_NUM_NEIGHBORS).long()
        bond_graph = torch.zeros(num_bonds, MAX_NUM_NEIGHBORS).long()

        for atom_idx in range(total_atoms):
            for neighbor_idx, bond_idx in enumerate(in_bonds[atom_idx]):
                atom_graph[atom_idx, neighbor_idx] = bond_idx

        for bond_idx_1 in range(1, num_bonds):
            x, y = all_bonds[bond_idx_1]
            for neighbor_idx, bond_idx_2 in enumerate(in_bonds[x]):
                # given the bond (x, y), don't consider the same
                # bond again i.e. (y, x)
                if all_bonds[bond_idx_2][0] != y:
                    bond_graph[bond_idx_1, neighbor_idx] = bond_idx_2

        return atom_feature_matrix, bond_feature_matrix, atom_graph, bond_graph, scope
