from copy import deepcopy

from rdkit import Chem
from chemutils import get_kekulized_mol_from_smiles, get_cluster_mol, get_smiles, enum_assemble


class MolJuncTreeNode(object):
    """
    This class, constructs a "cluster-node" for the junction tree, given a cluster (list of atom idx) in the original molecule.
    """

    _global_node_count = 0

    def __init__(self, smiles, cluster=[]):
        """
        Constructor for the class.

        Args:
        smiles: SMILES representation of the cluster
        cluster: List of atom_idx of atoms in this cluster

        Returns:
            The corresponding MolJuncTreeNode object.
        """

        # SMILES representation of the cluster
        self.smiles = smiles

        # get the molecule corresponding to this SMILES representation
        self.mol = get_kekulized_mol_from_smiles(self.smiles)

        # do a deepcopy to create a new list object
        self.cluster = deepcopy(cluster)

        # list to store neighbor nodes
        self.neighbors = []

        self.global_idx = MolJuncTreeNode._global_node_count

        MolJuncTreeNode._global_node_count += 1

    def add_neighbor(self, neighbor_node):
        """
        This method, add neighbor nodes for this given node in the "cluster-graph".

        Args:
            neighbor_node: The MolTreeJuncNode to be added as neighbor.
        """
        self.neighbors.append(neighbor_node)

    def recover(self, original_mol):
        """
        This method, given the original molecule, of which this node's cluster is a part of,
        reconstruct the molecular fragment, consisting of this particular cluster and all its neighbor clusters

        Args:
            original_mol: The original molecule, of which is cluster is a part of.
        """
        cluster = []
        cluster.extend(self.cluster)

        # atomMapNum is used as a cluster label
        # we set the atomMapNum for all the atoms belonging to a particular cluster,
        # to the cluster idx of that particular cluster

        # if this node is not a leaf node,
        # then for all the atoms of this cluster in the original molecule,
        # set the AtomMapNum to the id of this "cluster-node"
        if not self.is_leaf:
            for atom_idx in self.cluster:
                atom = original_mol.GetAtomWithIdx(atom_idx)
                atom.SetAtomMapNum(self.nid)

        # similarly, for all the neighbor nodes,
        # for all the atom of the "neighbor cluster" in the original molecule,
        # set the AtomMapNum to the id of these "neighbor cluster-nodes"
        for neighbor_node in self.neighbors:
            cluster.extend(neighbor_node.cluster)

            # # leaf node, no need to mark
            if neighbor_node.is_leaf:
                continue
            for atom_idx in neighbor_node.cluster:
                # allow singleton node override the atom mapping

                # if the atom is not in current node's cluster i.e. it is not a shared atom,
                # then set the AtomMapNum to node_id of neighbor node
                # or, if this atom corresponds to a singleton cluster, then allow this
                # atom to override current "cluster-node's" node_id
                if atom_idx not in self.cluster or len(neighbor_node.cluster) == 1:
                    atom = original_mol.GetAtomWithIdx(atom_idx)
                    atom.SetAtomMapNum(neighbor_node.nid)

        # a mega-cluster, corresponding to combination of current node's and its neighbors' clusters
        mega_cluster = list(set(cluster))

        # obtain the molecular fragment corresponding to this mega-cluster
        label_mol = get_cluster_mol(original_mol, mega_cluster)

        # obtain the corresponding SMILES representation for this molecular fragment
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))

        # obtain the corresponding molecular representation from the SMILES representation
        self.label_mol = get_kekulized_mol_from_smiles(self.label)

        # reset atom mapping to 0 for all the atoms of the original molecule
        for atom_idx in mega_cluster:
            original_mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(0)

        return self.label

    def assemble(self):
        """
        This function, given the current "cluster-node" in the "cluster-graph" and its "neighbor cluster-nodes",
        returns all the possible molecular attachment configurations
        of this node's cluster with its neighbor nodes' clusters.
        """
        # get the neighbors for this "cluster-node" which are not singleton clusters i.e. contain only one atom
        neighbors = [neighbor for neighbor in self.neighbors if neighbor.mol.GetNumAtoms() > 1]

        # sort the neighbor nodes of the "cluster-graph" in descending order of number of atoms
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)

        # obtain all the singleton neighbor "cluster-nodes" of this "cluster-node", in the "cluster-graph"
        singletons = [neighbor for neighbor in self.neighbors if neighbor.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        # obtain all possible candidate molecular attachment configurations,
        # corresponding to all possible valid combination of this cluster
        # and its neighbors
        candidates = enum_assemble(self, neighbors)

        if len(candidates) > 0:
            # SMILES, molecules
            self.candidates, self.candidate_mols, _ = zip(*candidates)
            self.candidates = list(self.candidates)
            self.candidate_mols = list(self.candidate_mols)
        else:
            self.candidates = []
            self.candidate_mols = []

