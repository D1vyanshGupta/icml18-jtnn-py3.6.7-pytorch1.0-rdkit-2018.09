from collections import deque

import torch
import torch.nn as nn

from GraphGRU import GraphGRU
from nnutils import create_var, index_select_ND

MAX_NUM_NEIGHBORS = 8

class JTNNEncoder(nn.Module):
    """
    Description: This module implements the encoder for the Junction Trees.
    """

    def __init__(self, hidden_size, depth, embedding=None):
        """
        The constructor for the class.

        Args:
            hidden_size: int
                The dimension of the hidden message vectors.

            depth: int
                The number of timesteps for which to implement the message passing.

            embedding: torch.embedding
                The embedding space for obtaining embeddings of atom features vectors.
        """
        # invoke the superclass constructor
        super(JTNNEncoder, self).__init__()

        # size of "hidden message vectors"
        self.hidden_size = hidden_size

        # number of timesteps to implement the message passing for
        self.depth = depth

        # embedding space for obtaining embeddings of atom features vectors.
        self.embedding = embedding

        # neural network to produce the output embedding
        # single hidden layer, followed by ReLU activation
        self.outputNN = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )

        # GRU for aggregation of message vectors from child nodes
        self.GRU = GraphGRU(hidden_size, hidden_size, self.depth)

    # def forward(self, root_batch):
    #     """
    #     Args:
    #         root_batch:
    #         list / batch of root nodes of the corresponding junction trees of the batch of molecules.
    #
    #     Returns:
    #         h: Dictionary containing hidden message vectors for all the edges, of all the junction trees, across the
    #            entire training dataset.
    #         root_vecs; Root vectors for all the junction trees, across the entire training dataset.
    #     """
    #     # list to store bottom-up & top-down traversal order for each junction tree
    #
    #     traversal_order_list = []
    #
    #     # get the traversal order for each junction tree given root node
    #     for root in root_batch:
    #         traversal_order = self.get_bottom_up_top_down_traversal_order(root)
    #         traversal_order_list.append(traversal_order)
    #
    #     # dictionary for storing hidden messages along various edges
    #     h = {}
    #
    #     max_iter = max([len(traversal_order) for traversal_order in traversal_order_list])
    #
    #     # if no messages from any neighbor node, then use this vector of zeros as
    #     # neighbor message vector
    #     padding = create_var(torch.zeros(self.hidden_size), False)
    #
    #     for iter in range(max_iter):
    #
    #         edge_tuple_list = []
    #
    #         for traversal_order in traversal_order_list:
    #             # keep appending traversal orders for a particular depth level,
    #             # from a given traversal_order list,
    #             # until the list is not empty
    #             if iter < len(traversal_order):
    #                 edge_tuple_list.extend(traversal_order[iter])
    #
    #         # for each edge, list of wids (word_idx corresponding to the cluster vocabulary item) of the current node.
    #         cur_x = []
    #
    #         # hidden messages for the current iteration, for the junction trees, across the entire dataset.
    #         cur_h_nei = []
    #
    #         for node_x, node_y in edge_tuple_list:
    #             x, y = node_x.idx, node_y.idx
    #             # wid is the index of the SMILES string, corresponding to the vocabulary cluster
    #             # of the node
    #             cur_x.append(node_x.wid)
    #
    #             # hidden messages from predecessor neighbor nodes of x, to x
    #             h_nei = []
    #
    #             for node_z in node_x.neighbors:
    #                 z = node_z.idx
    #                 if z == y:
    #                     continue
    #                 # hidden message from predecessor neighbor node z to node x
    #                 h_nei.append(h[(z, x)])
    #
    #             # each node can have at most MAX_NUM_NEIGHBORS(= 8) neighbors
    #             # thus we have a fixed construct of 8 message vectors
    #             # if a node doesn't receive messages from all of 8 neighbors,
    #             # then we set these message vectors to the zero vector
    #             pad_len = MAX_NUM_NEIGHBORS - len(h_nei)
    #             h_nei.extend([padding] * pad_len)
    #
    #             # append the chunk of hidden message vectors from neighbors to the cur_h_nei list
    #             # for batch operation
    #             cur_h_nei.extend(h_nei)
    #
    #         # for each wid in the list, get the corresponding word embedding
    #         cur_x = create_var((torch.tensor(cur_x)))
    #         cur_x = self.embedding(cur_x)
    #
    #         # hidden edge message vector for this iteration, for all the junction trees, across the entire
    #         # training dataset.
    #         cur_h_nei = torch.cat(cur_h_nei, dim=0).view(-1, MAX_NUM_NEIGHBORS, self.hidden_size)
    #
    #         # calculate the hidden messages for the next iteration, using the GRU operation.
    #         new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
    #
    #         # put the hidden messages for the next iteration, in the dictionary.
    #         for idx, edge in enumerate(edge_tuple_list):
    #             x, y = edge[0].idx, edge[1].idx
    #             h[(x, y)] = new_h[idx]
    #
    #     # evaluate root vectors encoding the structure for all the junction tress, across the entire training dataset.
    #     root_vecs = self.evaluate_root_vecs(root_batch, h, self.embedding, self.W)
    #
    #     return h, root_vecs

    # def get_bottom_up_top_down_traversal_order(self, root):
    #     """
    #     This method, gets the bottom-up and top-down traversal order for tree message passing purposes.
    #
    #     * node.idx is the id of the node across all nodes, of all junction trees, for all molecules of the dataset.
    #
    #     Args:
    #     root: Root of junction tree of a molecule in the training dataset.
    #
    #     Returns:
    #         traversal_order: List of lists of tuples. Each sublist of tuples corresponds to a depth of junction tree.
    #                         Each tuple corresponds to an edge along which message passing occurs.
    #     """
    #
    #     # FIFO queue for BFS traversal
    #     fifo_queue = deque([root])
    #
    #     # set to keep track of visited nodes
    #     visited = set([root.idx])
    #
    #     # root node is at zeroth depth
    #     root.depth = 0
    #
    #     # list to store appropriate traversal order
    #     top_down, bottom_up = [], []
    #
    #     while len(fifo_queue) > 0:
    #         # pop node from front of the queue
    #         x = fifo_queue.popleft()
    #
    #         # traverse the neighbors
    #         for y in x.neighbors:
    #             if y.idx not in visited:
    #                 fifo_queue.append(y)
    #
    #                 visited.add(y.idx)
    #
    #                 y.depth = x.depth + 1
    #
    #                 if y.depth > len(top_down):
    #                     # have a separate sublist for every depth
    #                     top_down.append([])
    #                     bottom_up.append([])
    #
    #                 top_down[y.depth - 1].append((x, y))
    #                 bottom_up[y.depth - 1].append((y, x))
    #
    #     # first we implement bottom-up traversal and then top-down traversal
    #     traversal_order = bottom_up[::-1] + top_down
    #
    #     return traversal_order
    #
    # def evaluate_root_vecs(self, root_batch, h, embedding, W):
    #     """
    #     This method, returns the hidden vectors for the root nodes for all the junction trees, across the entire
    #     training dataset.
    #
    #     Args:
    #         root_batch: list / batch of root nodes of the corresponding junction trees of the batch of molecules
    #         h: dictionary of hidden messages along all the edges, of all the junction trees, across the entire training dataset.

    #         embedding: embedding space for vocabulary composition
    #         W: weight matrix for calculating the hidden vectors for the root nodes of the junction trees, across the entire
    #         training dataset.
    #
    #     Returns:
    #         root_vecs: Hidden vectors for the root nodes of all the junction trees, across all the molecules of the
    #         training dataset.
    #     """
    #
    #     # for each root node, store the idx of the corresponding cluster vocabulary item.
    #     x_idx = []
    #
    #     # list to store lists of hidden edge message vectors from neighbors to root, for all root
    #     # nodes in the root_batch
    #     h_nei = []
    #
    #     hidden_size = embedding.embedding_dim
    #
    #     padding = create_var(torch.zeros(hidden_size), False)
    #
    #     for root in root_batch:
    #         x_idx.append(root.wid)
    #
    #         # list to store hidden edge messages from neighbors of each root node
    #         hidden_edge_messages = [h[(node_y.idx, root.idx)] for node_y in root.neighbors]
    #
    #         # each node can have at most MAX_NUM_NEIGHBORS(= 8 ) neighbors
    #         # thus we have a fixed construct of 8 message vectors
    #         # if a node doesn't receive messages from all of 8 neighbors,
    #         # then we set these message vectors to the zero vector
    #         pad_len = MAX_NUM_NEIGHBORS - len(hidden_edge_messages)
    #         hidden_edge_messages.extend([padding] * pad_len)
    #         h_nei.extend(hidden_edge_messages)
    #
    #     h_nei = torch.cat(h_nei, dim=0).view(-1, MAX_NUM_NEIGHBORS, hidden_size)
    #
    #     sum_h_nei = h_nei.sum(dim=1)
    #
    #     x_vec = create_var(torch.LongTensor(x_idx))
    #
    #     x_vec = embedding(x_vec)
    #
    #     root_vecs = torch.cat([x_vec, sum_h_nei], dim=1)
    #     return nn.ReLU()(W(root_vecs))

    def forward(self, node_wid_list, edge_node_idx_list, node_message_graph, mess_adjacency_graph, scope):
        """
        Args:
            node_wid_list: torch.LongTensor() (shape: num_edges)
                The list of wids i.e. idx of the corresponding cluster vocabulary item, for the initial node of each edge.

            edge_node_idx_list: torch.LongTensor() (shape: num_edges)
                The list of idx of the initial node of each edge.

            node_message_graph: torch.LongTensor (shape: num_nodes x MAX_NUM_NEIGHBORS)
                For each node, the list of idxs of all "inward hidden edge message vectors" for purposes of node feature aggregation.

            mess_adjacency_graph: torch.LongTensor (shape: num_edges x MAX_NUM_NEIGHBORS)
                For each edge, the list of idxs of all "inward hidden edge message vectors" for purposes of node feature aggregation.

            scope: List[Tuple(int, int)]
                The list to store tuples of (start_idx, len) to segregate all the node features, for a particular junction-tree.

            mess_dict: Dict{Tuple(int, int): int}
                The dictionary mapping edge in the form (x.idx, y.idx) to idx of message.

        Returns:
            tree_vecs: torch.tensor (shape: batch_size x hidden_size)
                The hidden vectors for the root nodes, of all the junction-trees, across the entire dataset.

        """
        # create PyTorch variables
        node_wid_list = create_var(node_wid_list)
        edge_node_idx_list = create_var(edge_node_idx_list)
        node_message_graph = create_var(node_message_graph)
        mess_adjacency_graph = create_var(mess_adjacency_graph)

        # hidden vectors for all the edges
        messages = create_var(torch.zeros(mess_adjacency_graph.size(0), self.hidden_size))

        # obtain node feature embedding
        node_feature_embeddings = self.embedding(node_wid_list)

        # for each edge obtain the embedding for the initial node
        initial_node_features = index_select_ND(node_feature_embeddings, 0, edge_node_idx_list)

        # obtain the hidden vectors for all the edges using GRU
        messages = self.GRU(messages, initial_node_features, mess_adjacency_graph)

        # for each node, obtain all the neighboring message vectors
        node_neighbor_mess_vecs = index_select_ND(messages, 0, node_message_graph)

        # for each node, sum up all the neighboring message vectors
        node_neighbor_mess_vecs_sum = node_neighbor_mess_vecs.sum(dim=1)

        # for each node, concatenate the node embedding feature and the sum of hidden neighbor message vectors
        node_vecs_synaptic_input = torch.cat([node_feature_embeddings, node_neighbor_mess_vecs_sum], dim=-1)

        # apply the neural network layer
        node_vecs = self.outputNN(node_vecs_synaptic_input)

        # list to store feature vectors of the root node, for all the junction-trees, across the entire dataset
        root_vecs = []

        for start_idx, _ in scope:
            # root node is the first node in the list of nodes of a juncion-tree by design
            root_vec = node_vecs[start_idx]
            root_vecs.append(root_vec)

        # stack the root tensors to form a 2-D tensor
        tree_vecs = torch.stack(root_vecs, dim=0)
        return tree_vecs, messages

    @staticmethod
    def tensorize(junc_tree_batch, cuda_device):
        """
        Args:
            junc_tree_batch: List[MolJuncTree]
                The list of junction-trees of all the molecular graphs in the dataset.
        """
        # list to store junction-tree nodes, for all junction-trees, across the entire dataset
        node_batch = []

        # list to store tuples of (start_idx, len) to segregate all the node features, for a particular junction-tree
        scope = []
        for junc_tree in junc_tree_batch:

            # starting idx of collection of nodes, for this junction-tree
            start_idx = len(node_batch)

            # the number of nodes in this junction-tree
            _len = len(junc_tree.nodes)

            # scope for this junction-tree
            scope_tuple = (start_idx, _len)
            scope.append(scope_tuple)

            # append the nodes of this junction-tree to list
            node_batch.extend(junc_tree.nodes)

        return JTNNEncoder.tensorize_nodes(node_batch, scope, cuda_device)

    @staticmethod
    def tensorize_nodes(node_batch, scope, cuda_device):
        """
        Args:
            node_batch: List[MolJuncTreeNode]
                The list of junction-tree nodes, of all junction-trees, across the entire dataset.

            scope: List[Tuple(int, int)]
                The list to store tuples of (start_idx, len) to segregate all the node features, for a particular junction-tree.

        Returns:
            node_wid_list: torch.LongTensor() (shape: num_edges)
                The list of wids i.e. idx of the corresponding cluster vocabulary item, for the initial node of each edge.

            edge_node_idx_list: torch.LongTensor() (shape: num_edges)
                The list of idx of the initial node of each edge.

            node_message_graph: torch.LongTensor (shape: num_nodes x MAX_NUM_NEIGHBORS)
                For each node, the list of idxs of all "inward hidden edge message vectors" for purposes of node feature aggregation.

            mess_adjacency_graph: torch.LongTensor (shape: num_edges x MAX_NUM_NEIGHBORS)
                For each edge, the list of idxs of all "inward hidden edge message vectors" for purposes of node feature aggregation.

            scope: List[Tuple(int, int)]
                The list to store tuples of (start_idx, len) to segregate all the node features, for a particular junction-tree.

            mess_dict: Dict{Tuple(int, int): int}
                The dictionary mapping edge in the form (x.idx, y.idx) to idx of message.

        """
        # list to store all edges / messages, for all the junction-trees,
        # across the entire dataset.

        # ensure messages are always 1-indexed
        messages = [None]

        # dictionary mapping edge in the form (x.idx, y.idx) to idx of message
        mess_dict = {}

        # list to store wids of all the nodes
        # for all the junction-trees, across the entire dataset.
        node_wid_list = []

        for x in node_batch:
            # append the wid or the node i.e. the idx of the cluster vocabulary item, to which this node corresponds to
            node_wid_list.append(x.wid)

            for y in x.neighbors:
                mess_dict[(x.idx, y.idx)] = len(messages)
                messages.append((x, y))

        # list of lists, to store the idxs of all the "inward" messages, for all the nodes,
        # of all the junction-tress, across the entire dataset.
        node_message_graph = [[] for idx in range(len(node_batch))]

        # list of lists, to store idx of messages from "inward-edges", for all the edges,
        # of all the junction-trees, across the entire dataset.
        mess_adjacency_graph = [[] for idx in range(len(messages))]

        # list to store the idx of initial node, for all the edges,
        # of all the junction-trees, across the entire dataset.
        edge_node_idx_list = [0] * len(messages)

        # iterate through the edges (x, y)
        for x, y in messages[1:]:
            # retrieve the idx of the message vector for edge (x, y)
            mess_idx_1 = mess_dict[(x.idx, y.idx)]

            # for the edge (x, y), node x is the initial node
            edge_node_idx_list[mess_idx_1] = x.idx

            # for the node y, the message from edge (x, y) will be used in aggregation procedure.
            node_message_graph[y.idx].append(mess_idx_1)

            for z in y.neighbors:
                # ignore, if the neighbor node is x
                if z.idx == x.idx:
                    continue
                # for all edges of the form, (y, z), edge (x, y) is an "inward edge"
                mess_idx_2 = mess_dict[(y.idx, z.idx)]
                mess_adjacency_graph[mess_idx_2].append(mess_idx_1)

        # the maximum number of message vectors from "inward edges" for all "nodes",
        # of all junction trees, across the entire dataset
        max_len = max([len(mess_idx_list) for mess_idx_list in node_message_graph] + [1])
        for mess_idx_list in node_message_graph:
            pad_len = max_len - len(mess_idx_list)
            mess_idx_list.extend([0] * pad_len)

        # the maximum number of message vectors from "inward edges" for all "edges",
        # of all junction trees, across the entire dataset
        max_len = max([len(mess_idx_list) for mess_idx_list in mess_adjacency_graph] + [1])
        for mess_idx_list in mess_adjacency_graph:
            pad_len = max_len - len(mess_idx_list)
            mess_idx_list.extend([0] * pad_len)

        node_message_graph = torch.LongTensor(node_message_graph)
        mess_adjacency_graph = torch.LongTensor(mess_adjacency_graph)
        edge_node_idx_list = torch.LongTensor(edge_node_idx_list)
        node_wid_list = torch.LongTensor(node_wid_list)

        node_message_graph.to(cuda_device)
        mess_adjacency_graph.to(cuda_device)
        edge_node_idx_list.to(cuda_device)
        node_wid_list.to(cuda_device)

        return (node_wid_list, edge_node_idx_list, node_message_graph, mess_adjacency_graph, scope), mess_dict
