import rdkit.Chem as Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

# given a node in the junction tree and it's neighbor nodes,
# this is the maximum number of molecular attachment configurations to be considered
MAX_MOL_CANDIDATES = 2000

def set_atom_map(mol, num=0):
    """
    Description: THhis function, given a molecule, sets the AtomMapNum of all atoms in the molecule, to the given num

    Args:
        mol: (object: rdkit)
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)

def get_kekulized_mol_from_smiles(smiles):
    """
    Description: This function, given the SMILES representation of a molecule,
    returns the kekulized molecule.

    Args:
        smiles: str
            SMILES representation of the molecule to be kekulized.

    Returns:
        mol: (object: rdkit)
            Kekulized representation of the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    """
    Description: This function, given a molecule, returns the SMILES representation,
    which encodes the kekulized structure of the molecule

    Args:
        mol: (object: rdkit)
            The molecule to be kekulized.

    Returns:
        SMILES: str
            SMILES representation, encoding the kekulized structure of the molecule
    """
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def decode_stereo(smiles2D):
    """
    Description: This function, given the SMILES representation of a molecule, encoding its 2D structure,
    gives the list of SMILES representation of all stereoisomers of the molecule, encoding their 3D structure

    Args:
        smiles2D: str
            SMILES representation, encoding its 2D structure,

    Returns:
        smiles3D: List[str]
            The list of SMILES representation of all stereoisomers of the molecule, encoding their 3D structure
    """

    # convert to molecular representation, from the SMILES representation
    mol = Chem.MolFromSmiles(smiles2D)

    # obtain all the stereoisomers of the molecule
    stereo_isomers = list(EnumerateStereoisomers(mol))
    stereo_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in stereo_isomers]

    # obtain SMILES representation of all stereoisomers, encoding their 3D structure
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in stereo_isomers]

    # in organic chemistry, nitrogen atoms are common chiral centers
    # thus, we get the idx of all nitrogen atoms that are chiral
    chiral_N_atoms = [atom.GetIdx() for atom in stereo_isomers[0].GetAtoms() if
                      int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == 'N']

    # if there are any chiral nitrogen centers, then we set their chiral tag to unspecified
    # because we are not currently dealing with chirality of nitrogen atoms
    if len(chiral_N_atoms) > 0:
        for mol in stereo_isomers:
            for idx in chiral_N_atoms:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D

def kekulize_mol(mol):
    """
    Description: This function, given a molecule, returns the kekulized representation of the same molecule.

    Args:
        mol: (object: rdkit)
            The molecule to be kekulized.

    Returns:
        mol: (object: rdkit)
            The kekulized molecule.
    """

    try:
        # obtain the SMILES representation
        smiles = get_smiles(mol)

        # obtain the kekulized molecule from the SMILES representation
        mol = get_kekulized_mol_from_smiles(smiles)
    except Exception as e:
        return None
    return mol

def deep_copy_atom(atom):
    """
    Description: This function, given an atom, returns a new atom, which is a "deep copy" of the given atom

    Args:
        atom: (object: rdkit)
            The atom to be copied.

    Returns:
        new_atom: (object: rdkit)
            New copy of the atom.
    """
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def deep_copy_mol(mol):
    """
    Description: This function, given a molecule, returns a new atom, which is a "deep copy" of the given molecule

    Args:
        mol: (object: rdkit)
            The molecule to be copied.

    Returns:
        new_mol: (object: rdkit)
            New copy of the molecule.
    """
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = deep_copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_cluster_mol(original_mol, cluster):
    """
    Description: This function, given the original molecule, and a cluster of atoms,
    returns the molecular fragment of the original molecule, corresponding to this cluster of atoms

    Args:
        original_mol: (object: rdkit)
            The original molecule, of which this cluster is a part of.

        cluster: List[int]
            List of atom_idx in this cluster

    Returns:
        mol: (object: rdkit)
            The valid molecular fragment corresponding to the given cluster of atoms.
    """

    # get the SMILES representation of the given cluster of atoms in this molecule
    smiles = Chem.MolFragmentToSmiles(original_mol, cluster, kekuleSmiles=True)

    # get the molecular fragment from the SMILES representation, corresponding to this cluster
    cluster_mol = Chem.MolFromSmiles(smiles, sanitize=False)

    # get a copy of the molecular fragment
    cluster_mol = deep_copy_mol(cluster_mol).GetMol()

    # obtain the kekulized representation of the molecular fragment
    cluster_mol = kekulize_mol(cluster_mol)

    return cluster_mol

def atom_equal(a1, a2):
    """
    Description: This function, given two atoms, checks if they have the same symbol and the same formal charge.

    Args:
        a1: (object: rdkit)
            The first atom

        a2: (object: rdkit)
            The second atom

    Returns:
        Boolean:
            Whether the two atoms have the same symbol and formal charge.
    """
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

def ring_bond_equal(b1, b2, reverse=False):
    """
    Description: This function, given two ring bonds, checks if they are the same or not.

    * bond type not considered because all bonds are aromatic i.e. ring bonds

    Args:
        b1: (object: rdkit)
            The first bond.

        b2: (object: rdkit)
            The second bond.

        reverse: Boolean
            Whether b2 has be to checked in reverse for equality.
    Returns:
         Boolean:
            Whether the bonds are same or not
    """
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])

def attach_mols(ctr_mol, neighbors, prev_nodes, neighbor_atom_map_dict):
    """
    Description: This function, given the center / current molecular fragment, neighbor nodes and atom_map_dict,
    constructs and returns the molecular attachment configuration as encoded in the atom_map_dict

    Args:
        ctr_mol: (object: rdkit)
            The center / current molecular fragment

        neighbors: List[MolJuncTreeNode]
            The list of neighbor nodes in the junction tree of that node, to which the center / current molecular fragment
            corresponds to.

        prev_nodes: List[MolJuncTreeNode]
            The list of nodes, already used in the center / current molecular fragment.

        neighbor_atom_map_dict: Dict{int: Dict{int: int}}
            A dictionary mapped to each neighbor node. For each neighbor node, the mapped dictionary
            further maps the atom idx of atom in neighbor node's cluster to the atom idx of atom in center / current molecular fragment.

    Returns:
        ctr_mol: (object: rdkit)
            The molecule attachment configuration as specified.
    """
    # nids of nodes previously used in the center / current molecular fragment
    prev_nids = [node.nid for node in prev_nodes]

    for neighbor_node in prev_nodes + neighbors:
        # obtain the neighbor node's node idx and the corresponding molecular fragment
        neighbor_id, neighbor_mol = neighbor_node.nid, neighbor_node.mol

        # obtain the atom_map corresponding to the atoms of this neighbor node's molecular fragment
        atom_map = neighbor_atom_map_dict[neighbor_id]

        for atom in neighbor_mol.GetAtoms():
            # if the atoms neighbor node's molecular fragment are not already present in the center / current
            # molecular fragment, then add them
            if atom.GetIdx() not in atom_map:
                new_atom = deep_copy_atom(atom)
                atom_map[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

        # if the neighbor node corresponds to a "singleton-cluster"
        if neighbor_mol.GetNumBonds() == 0:
            neighbor_atom = neighbor_mol.GetAtomWithIdx(0)
            ctr_atom = ctr_mol.GetAtomWithIdx(atom_map[0])
            ctr_atom.SetAtomMapNum(neighbor_atom.GetAtomMapNum())

        # if the neighbor node corresponds to either a "ring-cluster" or a "bond-cluster"
        else:
            for bond in neighbor_mol.GetBonds():
                bond_begin_atom = atom_map[bond.GetBeginAtom().GetIdx()]
                bond_end_atom = atom_map[bond.GetEndAtom().GetIdx()]
                if ctr_mol.GetBondBetweenAtoms(bond_begin_atom, bond_end_atom) is None:
                    ctr_mol.AddBond(bond_begin_atom, bond_end_atom, bond.GetBondType())
                # father node overrides
                elif neighbor_id in prev_nids:
                    ctr_mol.RemoveBond(bond_begin_atom, bond_end_atom)
                    ctr_mol.AddBond(bond_begin_atom, bond_end_atom, bond.GetBondType())
    return ctr_mol

def local_attach(ctr_mol, neighbors, prev_nodes, atom_map):
    """
    Description: This function, given the center / current molecular fragment, the current atom_map and multiple neighbor nodes,
    returns the molecular attachment configuration as encoded in the given atom_map.

    Args:
        ctr_mol: (object: rdkit)
            The center / current molecular fragment

        neighbors: List[MolJuncTreeNode]
            The list of neighbor nodes in the junction tree of that node, to which the center / current molecular fragment
            corresponds to.

        prev_nodes: List[MolJuncTreeNode]
            The list of nodes, already used in the center / current molecular fragment.

        atom_map: List[Tuple(int, int, int)]
            The atom_map encoding information about how the clusters corresponding to the neighbor clusters
            are attached to the current / center molecular fragment.

    * An atom_map, constructed with respect to a center / curent cluster, is a list of tuples of the form
    (neighbor_node.nid, idx of atom in center / current molecule, idx of atom in neighbor node's molecular fragment)

    Returns:
        ctr_mol: (object: rdkit)
            The molecule attachment configuration as specified in the atom_map.
    """
    ctr_mol = deep_copy_mol(ctr_mol)
    neighbor_atom_map_dict = {neighbor.nid: {} for neighbor in prev_nodes + neighbors}

    for neighbor_id, ctr_atom_idx, neighbor_atom_idx in atom_map:
        neighbor_atom_map_dict[neighbor_id][neighbor_atom_idx] = ctr_atom_idx

    ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, neighbor_atom_map_dict)
    return ctr_mol.GetMol()

def enum_attach(ctr_mol, neighbor_node, atom_map, singletons):
    """
    Description: This function, given the center / current molecular fragment, the current atom_map and neighbor node,
    enumerates all possible attachment configurations of the current molecular fragment
    with the neighbor node's molecular fragment.

    * An atom_map, constructed with respect to a center / curent cluster, is a list of tuples of the form
    (neighbor_node.nid, idx of atom in center / current molecule, idx of atom in neighbor node's molecular fragment).

    Args:
        ctr_mol: (object: rdkit)
            The center / current molecular fragment

        neighbors: List[MolJuncTreeNode]
            The list of neighbor nodes in the junction tree of that node, to which the center / current molecular fragment
            corresponds to.

        prev_nodes: List[MolJuncTreeNode]
            The list of nodes, already used in the center / current molecular fragment.

        atom_map: List[Tuple(int, int, int)]
            The atom_map encoding information about how the clusters corresponding to the neighbor clusters
            are attached to the current / center molecular fragments.

        singletons: List[int]
            The list of atom_idx of those atoms, which correspond to singleton clusters.

    Returns:
        att_confs: List[List[Tuple(int, int, int)]]
         The list of atom_maps corresponding to all possible attachment configurations.
    """

    # obtain the neighbor node's molecular fragment and node id
    neighbor_mol, neighbor_idx = neighbor_node.mol, neighbor_node.nid

    # list for storing all possible attachment configurations
    att_confs = []

    # exclude atoms corresponding to "singleton-clusters" from consideration
    black_list = [atom_idx for neighbor_id, atom_idx, _ in atom_map if neighbor_id in singletons]
    ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetIdx() not in black_list]

    ctr_bonds = [bond for bond in ctr_mol.GetBonds()]

    # if the neighbor node corresponds to a "singleton-cluster"
    if neighbor_mol.GetNumBonds() == 0:
        # obtain the only atom of this singleton cluster
        neighbor_atom = neighbor_mol.GetAtomWithIdx(0)

        # obtain the idx of all the atoms that have already been used in the current / center molecular fragment
        used_list = [atom_idx for _, atom_idx, _ in atom_map]

        for atom in ctr_atoms:
            if atom_equal(atom, neighbor_atom) and atom.GetIdx() not in used_list:
                new_atom_map = atom_map + [(neighbor_idx, atom.GetIdx(), 0)]
                att_confs.append(new_atom_map)

    # if the neighbor node corresponds to a simple "bond cluster"
    elif neighbor_mol.GetNumBonds() == 1:
        # obtain the only bond of the neighbor node's molecular fragment
        bond = neighbor_mol.GetBondWithIdx(0)

        # obtain the bond valence
        bond_val = int(bond.GetBondTypeAsDouble())

        # obtain the beginning and ending atoms of the bond
        bond_begin_atom, bond_end_atom = bond.GetBeginAtom(), bond.GetEndAtom()

        for atom in ctr_atoms:
            # optimize if atom is carbon (other atoms may change valence)
            if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() < bond_val:
                continue

            if atom_equal(atom, bond_begin_atom):
                new_atom_map = atom_map + [(neighbor_idx, atom.GetIdx(), bond_begin_atom.GetIdx())]
                att_confs.append(new_atom_map)
            elif atom_equal(atom, bond_end_atom):
                new_atom_map = atom_map + [(neighbor_idx, atom.GetIdx(), bond_end_atom.GetIdx())]
                att_confs.append(new_atom_map)

    # the neighbor node corresponds to a "ring molecular fragment"
    else:
        # if only one atom is common between the center / current molecular fragment and the neighbor molecular fragment
        for ctr_atom in ctr_atoms:
            for neighbor_atom in neighbor_mol.GetAtoms():
                if atom_equal(ctr_atom, neighbor_atom):
                    # optimize if atom is carbon (other atoms may change valence)
                    if ctr_atom.GetAtomicNum() == 6 and ctr_atom.GetTotalNumHs() + neighbor_atom.GetTotalNumHs() < 4:
                        continue
                    new_atom_map = atom_map + [(neighbor_idx, ctr_atom.GetIdx(), neighbor_atom.GetIdx())]
                    att_confs.append(new_atom_map)

        # if the intersection between the center / current molecular fragment
        # and the neighbor molecular fragment is a bond
        if ctr_mol.GetNumBonds() > 1:
            for ctr_bond in ctr_bonds:
                for neighbor_bond in neighbor_mol.GetBonds():
                    if ring_bond_equal(ctr_bond, neighbor_bond):
                        new_atom_map = atom_map + [
                            (neighbor_idx, ctr_bond.GetBeginAtom().GetIdx(), neighbor_bond.GetBeginAtom().GetIdx()),
                            (neighbor_idx, ctr_bond.GetEndAtom().GetIdx(), neighbor_bond.GetEndAtom().GetIdx())]
                        att_confs.append(new_atom_map)

                    if ring_bond_equal(ctr_bond, neighbor_bond, reverse=True):
                        new_atom_map = atom_map + [
                            (neighbor_idx, ctr_bond.GetBeginAtom().GetIdx(), neighbor_bond.GetEndAtom().GetIdx()),
                            (neighbor_idx, ctr_bond.GetEndAtom().GetIdx(), neighbor_bond.GetBeginAtom().GetIdx())]
                        att_confs.append(new_atom_map)
    return att_confs

def enum_assemble(node, neighbors, prev_nodes=[], prev_atom_map=[]):
    """
    Description: This function, given a node in the junction tree and its neighbor nodes,
    returns all the possible molecular attachment configurations
    of this node's cluster to with its neighbor nodes' clusters.

    * atom_maps incorporate information about how the clusters corresponding to nodes in the "cluster-graph"
    are attached together to form a valid molecular fragment

    Arguments:
        node: (object: MolJuncTreeNode)
            the node in the junction tree, whose molecular attachment configurations have to be enumerated.

        neighbors: List[MolJuncTreeNode]
            The neighbors to be considered for molecular attachment.

        prev_nodes: List[MolJuncTreeNode]
            The nodes already considered for molecular attachment.

        prev_atom_map: List[Tuple(int, int, int)]
            the previous atom map encoding information about the molecular attachment configuration with previously used neighbors.

    Returns:
        all_attach_confs: List[Tuple(str, object: rdkit, List[Tuple(int, int, int)])]
            List of tuples of all possible valid attachment configurations, of the form (smiles, molecule, atom_map).
    """

    # list of all possible valid, molecular attachment configurations of given node with its neighbor nodes
    all_attach_confs = []

    # get those "cluster-nodes" from the "neighbor-nodes" list that are "singleton-clusters"
    singletons = [neighbor_node.nid for neighbor_node in neighbors + prev_nodes
                  if neighbor_node.mol.GetNumAtoms() == 1]

    # search for all possible molecular attachment configurations of this node's cluster to its neighbor nodes' clusters
    def search(cur_atom_map, depth):
        # upper limit on the number of molecular attachment configurations to be considered
        # for efficiency purposes
        if len(all_attach_confs) > MAX_MOL_CANDIDATES:
            return

        # if all the neighbor nodes have considered for building a molecular attachment configuration,
        # then append this attachment configuration to attachment configuration list
        if depth == len(neighbors):
            all_attach_confs.append(cur_atom_map)
            return

        # the next neighbor node which is to be attached to the current molecular fragment
        neighbor_node = neighbors[depth]

        # return the list of possible atom_maps that encode information
        # about how the above neighbor node can be attached to the current molecular fragment
        candidate_atom_map_list = enum_attach(node.mol, neighbor_node, cur_atom_map, singletons)

        # set for storing SMILES representations of candidate molecular fragment configurations
        candidate_smiles = set()

        # list for storing candidate atom_maps which encode the possible ways in which the
        # neighbor node can be attached to the current molecular fragment
        candidate_atom_maps = []

        for atom_map in candidate_atom_map_list:
            # obtain a candidate molecular fragment in which the above neighbor node cluster
            # has been attached to the current molecular fragment
            candidate_mol = local_attach(node.mol, neighbors[:depth + 1], prev_nodes, atom_map)

            # obtain a kekulized representation of this candidate molecular fragment
            candidate_mol = kekulize_mol(candidate_mol)

            if candidate_mol is None:
                continue

            # obtain the SMILES representation of this molecule
            smiles = get_smiles(candidate_mol)

            if smiles in candidate_smiles:
                continue

            # add the candidate SMILES string to the list
            candidate_smiles.add(smiles)

            # add the candidate atom_map to the list
            candidate_atom_maps.append(atom_map)

        # if no more candidates atom_maps are available,
        # i.e. no more valid chemical intermediaries are possible,
        # then stop searching
        if len(candidate_atom_maps) == 0:
            return

        # for each of the candidate atom_maps, search for more candidate atom_maps using more neighbor nodes
        for new_atom_map in candidate_atom_maps:
            search(new_atom_map, depth + 1)

    # search for candidate atom_maps with the previous atom_map
    search(prev_atom_map, 0)

    # set for storing SMILES representations of candidate molecular fragment configurations
    candidate_smiles = set()

    # list to store tuples of (SMILES representation, candidate molecular fragment, atom_map)
    candidates = []

    for atom_map in all_attach_confs:
        # obtain the candidate molecular fragment
        candidate_mol = local_attach(node.mol, neighbors, prev_nodes, atom_map)
        candidate_mol = Chem.MolFromSmiles(Chem.MolToSmiles(candidate_mol))

        # obtain the SMILES representation of this molecule
        smiles = Chem.MolToSmiles(candidate_mol)

        if smiles in candidate_smiles:
            continue

        # add the candidate SMILES string to the list
        candidate_smiles.add(smiles)

        # obtain a kekulized representation of the molecule
        Chem.Kekulize(candidate_mol)

        candidates.append((smiles, candidate_mol, atom_map))

    return candidates