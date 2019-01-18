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
            The bond for which the feature vector is to be constructed.

    Returns:
        torch.tensor (shape: BOND_FEATURE_DIM)
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

    def forward(self, atom_feature_matrix, bond_feature_matrix, atom_adjacency_graph, bond_adjacency_graph, scope):
        """
        Description: Implements the forward pass for encoding the original molecular subgraphs, for the graph encoding step. (Section 2.2)

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
             mol_vecs: torch.tensor (shape: batch_size x hidden_size)
                The encoding vectors for all the molecular graphs in the entire dataset.
        """
        # create PyTorch Variables
        atom_feature_matrix = create_var(atom_feature_matrix)
        bond_feature_matrix = create_var(bond_feature_matrix)
        atom_adjacency_graph = create_var(atom_adjacency_graph)
        bond_adjacency_graph = create_var(bond_adjacency_graph)

        static_messages = self.W_i(bond_feature_matrix)

        # apply ReLU activation for timestep, t = 0
        message = nn.ReLU()(static_messages)

        # implement message passing for timesteps, t = 1 to T (depth)
        for timestep in range(self.depth - 1):

            # obtain messages from all the "inward edges"
            neighbor_message_vecs = index_select_ND(message, 0, bond_adjacency_graph)

            # sum up all the "inward edge" message vectors
            neighbor_message_vecs_sum = neighbor_message_vecs.sum(dim=1)

            # multiply with the weight matrix for the hidden layer
            neighbor_message = self.W_h(neighbor_message_vecs_sum)

            # message at timestep t + 1
            message = nn.ReLU()(static_messages + neighbor_message)

        # neighbor message vectors for each node from the message matrix
        neighbor_message_vecs = index_select_ND(message, 0, atom_adjacency_graph)

        # neighbor message for each atom
        neighbor_message_atom_matrix = neighbor_message_vecs.sum(dim=1)

        # concatenate atom feature vector and neighbor hidden message vector
        atom_input_matrix = torch.cat([atom_feature_matrix, neighbor_message_atom_matrix], dim=1)
        atom_hidden_layer_synaptic_input = nn.ReLU()(self.W_o(atom_input_matrix))

        # list to store the corresponding molecule vectors for each molecule
        mol_vecs = []
        for start_idx, len in scope:
            # the molecule vector for molecule, is the mean of the hidden vectors for all the atoms of that
            # molecule
            mol_vec = atom_hidden_layer_synaptic_input[start_idx : start_idx + len].mean(dim=0)
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

    @staticmethod
    def tensorize(smiles_batch, cuda_device):
        """
        Description: This method, given a batch of SMILES representations,
        constructs the feature vectors for the corresponding molecules

        Args:
            smiles_batch: List[str]
                The batch of SMILES representations for the dataset.

        Returns:

            atom_feature_matrix: torch.tensor (shape: batch_size x ATOM_FEATURE_DIM)
                The matrix containing feature vectors, for all the atoms, across the entire dataset.

            bond_feature_matrix: torch.tensor (shape: batch_size x ATOM_FEATURE_DIM + BOND_FEATURE_DIM)
                The matrix containing feature vectors, for all the bonds, across the entire dataset.

            atom_adjacency_graph: torch.tensor (shape: num_atoms x MAX_NUM_NEIGHBORS(=6))
                For each atom, across the entire dataset, the idxs of all the neighboring atoms.

            bond_adjacency_graph: torch.tensor (shape: num_bonds x MAX_NUM_NEIGHBORS(=6))
                For each bond, across the entire dataset, the idxs of the "inward bonds", for purposes of message passing.

            scope: List[Tuple(int, int)]
                The list to store tuples (total_bonds, num_bonds), to keep track of all the atom feature vectors,
                belonging to a particular molecule.

        """
        # add zero-padding for the bond feature vector matrix
        padding = torch.zeros(ATOM_FEATURE_DIM + BOND_FEATURE_DIM)

        # ensure that the bond feature vectors are 1-indexed
        atom_feature_vecs, bond_feature_vecs = [], [padding]

        # ensure that the bonds are is 1-indexed

        # in_bonds, for a given atom_idx in a molecule, stores the list of idxs of all bonds in which it is present
        in_bonds, all_bonds = [], [(-1, -1)]

        # list to store tuples of (start_index, len) for atoms feature vectors of each molecule
        scope = []

        # each atom, for all molecules, over the entire dataset, is given an idx.
        total_atoms = 0

        for smiles in smiles_batch:
            # get the corresponding molecule from the SMILES representation
            mol = get_kekulized_mol_from_smiles(smiles)

            # get the number of atoms in the molecule
            num_atoms = mol.GetNumAtoms()

            for atom in mol.GetAtoms():
                atom_feature_vecs.append(get_atom_feature_vec(atom))
                in_bonds.append([])

            for bond in mol.GetBonds():
                bond_begin_atom = bond.GetBeginAtom()
                bond_end_atom = bond.GetEndAtom()

                x = bond_begin_atom.GetIdx() + total_atoms
                y = bond_end_atom.GetIdx() + total_atoms

                num_bonds = len(all_bonds)
                all_bonds.append((x, y))
                bond_feature_vecs.append(torch.cat([atom_feature_vecs[x], get_bond_feature_vec(bond)], 0))
                in_bonds[y].append(num_bonds)

                num_bonds = len(all_bonds)
                all_bonds.append((y, x))
                bond_feature_vecs.append(torch.cat([atom_feature_vecs[y], get_bond_feature_vec(bond)], 0))
                in_bonds[x].append(num_bonds)

            scope.append((total_atoms, num_atoms))
            total_atoms += num_atoms

        num_bonds = len(all_bonds)
        atom_feature_matrix = torch.stack(atom_feature_vecs, 0)
        bond_feature_matrix = torch.stack(bond_feature_vecs, 0)
        atom_adjacency_graph = torch.zeros(total_atoms, MAX_NUM_NEIGHBORS).long()
        bond_adjacency_graph = torch.zeros(num_bonds, MAX_NUM_NEIGHBORS).long()

        for atom_idx in range(total_atoms):
            for neighbor_idx, bond_idx in enumerate(in_bonds[atom_idx]):
                atom_adjacency_graph[atom_idx, neighbor_idx] = bond_idx

        for bond_idx_1 in range(1, num_bonds):
            x, y = all_bonds[bond_idx_1]
            for neighbor_idx, bond_idx_2 in enumerate(in_bonds[x]):
                # given the bond (x, y), don't consider the same
                # bond again i.e. (y, x)
                if all_bonds[bond_idx_2][0] != y:
                    bond_adjacency_graph[bond_idx_1, neighbor_idx] = bond_idx_2

        atom_feature_matrix.to(cuda_device)
        bond_feature_matrix.to(cuda_device)
        atom_adjacency_graph.to(cuda_device)
        bond_adjacency_graph.to(cuda_device)

        return (atom_feature_matrix, bond_feature_matrix, atom_adjacency_graph, bond_adjacency_graph, scope)




