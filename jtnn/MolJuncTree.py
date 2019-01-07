from rdkit import Chem
from collections import defaultdict
from scipy.sparse import csr_matrix
from MolJuncTreeNode import MolJuncTreeNode
from scipy.sparse.csgraph import minimum_spanning_tree
from chemutils import get_kekulized_mol_from_smiles, decode_stereo, get_cluster_mol, get_smiles, set_atom_map

# max weight of an edge in "cluster-graph" of any molecule
MAX_EDGE_WEIGHT = 100


class MolJuncTree(object):
    """
    Given the SMILES representation of a molecule,
    this class constructs a corresponding Junction Tree for the molecule.
    """

    def __init__(self, smiles):
        """
        The constructor for the MolJuncTree class.

        Args:
            smiles: SMILES representation of molecule

        Returns:
            MolJuncTree object for the corresponding molecule.

        """

        # SMILES representation for the molecule
        self.smiles = smiles

        # kekulized molecular representation
        self.mol = get_kekulized_mol_from_smiles(self.smiles)

        # obtain all stereoisomers for this molecule
        mol = Chem.MolFromSmiles(smiles)

        self.smiles2D = Chem.MolToSmiles(mol)

        # assert(self.smiles == self.smiles2D)

        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)

        assert(self.smiles2D == self.smiles3D)

        # obtain list of SMILES representation of all stereoisomers of the molecule, encoding their 3D structure
        self.stereo_candidates = decode_stereo(self.smiles2D)

        # obtain the clusters in the molecule and the adjacency list for the junction tree
        clusters, edges = self.cluster_decomposition()

        # list for storing the nodes of the junction tree
        self.nodes = []

        # idx for denoting the root of the junction tree
        root = 0

        # construct the nodes for the junction tree
        for idx, cluster in enumerate(clusters):
            # obtain the molecular fragment corresponding to the cluster
            cluster_mol = get_cluster_mol(self.mol, cluster)

            # instantiate a MolTreeNode corresponding to this cluster
            node = MolJuncTreeNode(get_smiles(cluster_mol), cluster)

            # append the node to the created list of nodes
            self.nodes.append(node)

            # if the atom with atom_idx equal to 0, in present in this cluster,
            # then denote this particular cluster as the root of the junction tree
            if min(cluster) == 0:
                root = idx

        # for each of the nodes of the junction tree, add neighbors,
        # based on the adjacency list obtained from the tree decomposition process
        for cluster_idx_1, cluster_idx_2 in edges:
            self.nodes[cluster_idx_1].add_neighbor(self.nodes[cluster_idx_2])
            self.nodes[cluster_idx_2].add_neighbor(self.nodes[cluster_idx_1])

        # if the root node has a cluster idx greater than 0, then swap it with the node having cluster_idx = 0
        if root > 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

        # set node_ids / nids for all the nodes
        for idx, node in enumerate(self.nodes):
            node.nid = idx + 1

            # for each of the non-leaf nodes,
            # for the atoms of the corresponding cluster,
            # we set the atomMapNum of these atoms,
            # to the node_id / nid of that node / cluster

            # leaf nodes have only 1 neighbor
            if len(node.neighbors) > 1:
                set_atom_map(node.mol, node.nid)

            node.is_leaf = (len(node.neighbors) == 1)

    def __len__(self):
        return len(self.nodes)

    def cluster_decomposition(self):
        """
        This function is used to find the molecular clusters, of which the molecule is composed of.
        * In this function, a cluster is simply a list of all the atoms, which are present in that cluster.
        There are three kinds of clusters:
            1. Non-Ring Bond Clusters: Corresponding to non-ring bonds. (2 atoms in the cluster)
            2. Ring Clusters: Corresponding to rings. (More than 4 atoms in the cluster)
            3. Singleton Clusters: Corresponding to single atoms. These specific atoms are at the
               intersection of three or more bonds (can be both ring and non-ring bonds).
               The "cluster-subgraph", consisting of all the "cluster-nodes" containing this atom,
               "would form a clique". Thus, in an effort to reduce the number of cycles in the
               "cluster-graph", the authors have introduced the notion of singleton clusters.
               These singleton clusters consist of only one atom. These specific atoms are common to
               3 or more clusters. In the "cluster-graph", an edge is created between the "singleton cluster nodes"
               and all the other "cluster-nodes" that contain the atom corresponding to that "singleton cluster".
               These edge are assigned a large weight, so as to ensure that these edges are included when a
               "maximum spanning tree" over the "cluster-graph" is being found.

        Returns:
            clusters: list of clusters, of which the molecule is composed of.
            edges: adjacency list for the "maximum spanning tree", over the "cluster-graph"
        """

        num_atoms = self.mol.GetNumAtoms()

        # if there is only one atom, the [0] is the only cluster and the adjacency matrix is an empty list
        if num_atoms == 1:
            return [[0]], []

        # list to store clusters making up the molecule
        clusters = []

        # all pairs of atoms in non-ring bonds correspond to a non-ring bond cluster
        for bond in self.mol.GetBonds():
            # only consider non-ring bonds
            if bond.IsInRing():
                continue

            begin_atom_idx = bond.GetBeginAtom().GetIdx()
            end_atom_idx = bond.GetEndAtom().GetIdx()

            bond_cluster = [begin_atom_idx, end_atom_idx]

            # append the bond cluster to the list of clusters
            clusters.append(bond_cluster)

        # obtain the clusters corresponding to rings in the molecule, using rdkit's GetSymmSSSR function
        rings_clusters = [list(ring) for ring in Chem.GetSymmSSSR(self.mol)]

        # include the list of rings in the list of clusters
        clusters.extend(rings_clusters)

        # for each atom, get the list of idx of all clusters, in which it is included
        # this is for constructing edges between clusters in the "cluster-graph", that share at least one atom
        atom_cluster_idx_map = {atom_idx : [] for atom_idx in range(num_atoms)}

        # cstr_idx: cluster_idx

        for cluster_idx in range(len(clusters)):
            for atom_idx in clusters[cluster_idx]:
                atom_cluster_idx_map[atom_idx].append(cluster_idx)

        # merge rings with having more than two atoms in common i.e. bridged compounds
        for cluster_idx_i in range(len(clusters)):
            # ignore clusters corresponding to non-ring bonds and single atoms
            if len(clusters[cluster_idx_i]) <= 2:
                continue

            for atom_idx in clusters[cluster_idx_i]:
                for cluster_idx_j in atom_cluster_idx_map[atom_idx]:

                    # merge ring clusters in order of i < j

                    # if len(clusters[cluster_idx_j]) <= 2 i.e. this clusters corresponds to a non-ring bond or single atoms,
                    # then don't merge

                    if cluster_idx_i >= cluster_idx_j or len(clusters[cluster_idx_j]) <= 2:
                        continue

                    # find the number of common atoms between the two clusters
                    inter = set(clusters[cluster_idx_i]) & set(clusters[cluster_idx_j])

                    # merge the rings, only if they have more than 2 atoms in common i.e. bridged compounds
                    if len(inter) > 2:
                        clusters[cluster_idx_i].extend(clusters[cluster_idx_j])
                        clusters[cluster_idx_i] = list(set(clusters[cluster_idx_i]))
                        clusters[cluster_idx_j] = []

        # remove empty clusters
        clusters = [cluster for cluster in clusters if len(cluster) > 0]

        # for each atom, get the list of all cluster_idx, in which it is included
        # this is for constructing edges between clusters in the "cluster-graph", that share at least one atom
        atom_cluster_idx_map = {atom_idx : [] for atom_idx in range(num_atoms)}
        for cluster_idx in range(len(clusters)):
            for atom_idx in clusters[cluster_idx]:
                atom_cluster_idx_map[atom_idx].append(cluster_idx)

        # build edges and singleton clusters

        # adjacency list of "cluster-graph"
        edges = defaultdict(int)

        # atoms are common between at least 3 clusters, correspond to "singleton clusters"
        # some examples are as follows:
        # 1. number of bonds > 2
        # 2. number of bonds = 2 and number of rings > 1
        # 3. number of bonds = 1 and number of rings = 2 (not considered here)
        # 4. number of rings > 2

        for atom_idx in range(num_atoms):
            # ignore if this atom already belongs to a singleton cluster
            if len(atom_cluster_idx_map[atom_idx]) <= 1:
                continue

            # list of idx, of all the clusters, in which this atom is included
            cluster_idx_list = atom_cluster_idx_map[atom_idx]

            # idx of clusters corresponding to non-ring bonds
            bonds_clusters = [cluster_idx for cluster_idx in cluster_idx_list if len(clusters[cluster_idx]) == 2]

            # idx of clusters corresponding to rings
            ring_clusters = [cluster_idx for cluster_idx in cluster_idx_list if len(clusters[cluster_idx]) > 4]

            # in general, if len(cluster_idx_list) >= 3, a singleton should be added,
            # but the case of 1 bond & 2 rings is not currently dealt with

            # "singleton cluster" candidates considered:
            # 1. number of bonds > 2
            # 2. number of bonds = 2 and number of rings > 1
            if len(bonds_clusters) > 2 or (len(bonds_clusters) == 2 and len(cluster_idx_list) > 2):
                # append a singleton cluster, corresponding to this particular atom
                singleton_cluster = [atom_idx]
                clusters.append(singleton_cluster)

                # obtain the idx of this new "singleton cluster"
                new_cluster_idx = len(clusters) - 1

                # create edges between this singleton cluster and all the clusters in which this atom is present
                for cluster_idx in cluster_idx_list:
                    edges[(cluster_idx, new_cluster_idx)] = 1

            # singleton candidates considered:
            # 1. number of rings > 2
            elif len(ring_clusters) > 2:
                # append a singleton cluster, corresponding to this particular atom
                singleton_cluster = [atom_idx]
                clusters.append(singleton_cluster)

                # obtain the idx of this new singleton cluster
                new_cluster_idx = len(clusters) - 1

                # create edges between this singleton cluster and all the clusters in which this atom is present
                for cluster_idx in cluster_idx_list:
                    edges[(cluster_idx, new_cluster_idx)] = MAX_EDGE_WEIGHT - 1

            # build edges between all the clusters that share at least one atom
            else:
                for cluster_idx_i in range(len(cluster_idx_list)):
                    for cluster_idx_j in range(cluster_idx_i + 1, len(cluster_idx_list)):
                        cluster_idx_1, cluster_idx_2 = cluster_idx_list[cluster_idx_i], cluster_idx_list[cluster_idx_j]

                        # find number of common atoms between the two clusters
                        inter = set(clusters[cluster_idx_1]) & set(clusters[cluster_idx_2])

                        # assign weight equal to number of common atoms, to the edge between the two clusters
                        if edges[(cluster_idx_1, cluster_idx_1)] < len(inter):
                            edges[(cluster_idx_1, cluster_idx_2)] = len(inter)

        # to obtain a maximum spanning tree of any given graph, we subtract the edge weights by
        # a large value, and then simply find a minimum spanning tree of this new graph with modified edge weights

        edges = [edge + (MAX_EDGE_WEIGHT - weight,) for edge, weight in edges.items()]

        if len(edges) == 0:
            return clusters, edges

        row, col, weight = zip(*edges)
        num_clusters = len(clusters)
        cluster_graph = csr_matrix((weight, (row, col)), shape=(num_clusters, num_clusters))

        # obtain the junction-tree for this molecule
        junc_tree = minimum_spanning_tree(cluster_graph)

        # obtain a sparse representation of this juntion-tree
        row, col = junc_tree.nonzero()
        edges = [(row[i], col[i]) for i in range(len(row))]

        # finally return the list of clusters and the adjacency list for the corresponding junction tree
        return clusters, edges

    def recover(self):
        """
        This function is used to recover all of the nodes in the junction tree,
        i.e. for each of the nodes, reconstruct the molecular fragment consisting of that node and its neighbors
        """
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        """
        Assemble all of the nodes in the junction tree,
        i.e. for each node, obtain all possible molecular attachment configurations of that node and its neighbors
        """
        for node in self.nodes:
            node.assemble()

# if __name__ == "__main__":
#     import sys
#     lg = rdkit.RDLogger.logger()
#     lg.setLevel(rdkit.RDLogger.CRITICAL)
#
#     idx = 0
#
#     cset = set()
#     for i,line in enumerate(sys.stdin):
#         if idx == 10:
#             break
#         smiles = line.split()[0]
#         mol = MolJuncTree(smiles)
#         for c in mol.nodes:
#             cset.add(c.smiles)
#         idx += 1
#
#     with open('vocab_test.txt', 'w') as file:
#         for x in cset:
#             file.write(x + "\n")

