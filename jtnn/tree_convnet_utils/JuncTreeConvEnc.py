from collections import deque

import torch
import torch.nn as nn

from tree_convnet_utils.JuncTreeConvEncLayer import ConvEncLayer

class JuncTreeConvEnc(nn.Module):
    """
    Description: This class implements a graph convolutional encoder, for junction trees of molecular graphs
    """

    def __init__(self, vocab, hidden_size, embedding, num_layers):
        """
        Description: Constructor for the class.

        Args:
            vocab: List[MolJuncTreeNode]
                The list of cluster vocabulary items for the given dataset.

            hidden_size: int
                The dimension of the hidden edge vectors.

            embedding: torch.embedding
                The embedding latent space for encoding the features of a given cluster node in a junction-tree.

            num_layers: int
                The number of convolutional layers in the graph convolutional encoder.
        """
        # invoke superclass constructor
        super(JuncTreeConvEnc, self).__init__()

        # size of the hidden feature vectors for nodes in the junction-tree
        self.hidden_size = hidden_size

        # cluster vocabulary over the entire dataset
        self.vocab = vocab

        if embedding is None:
            # setup embedding space for encoding the features of given cluster node in the junction tree
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        # number of convolutional layers in the graph encoder
        self.num_layers = num_layers

        # list to store convolutional layers for the encoder
        convnet_layers = []

        # instantiate convnet layers for the encoder
        for layer_num in range(num_layers):
            convnet_layer = ConvEncLayer(self.hidden_size)
            convnet_layers.append(convnet_layer)

        # set convnet layers for the encoder
        self.convnet_layers = nn.ModuleList(convnet_layers)

    def forward(self, junc_tree_batch):
        """
        Args:
            junc_tree_batch: List[MolJuncTree]
                The list of junction-trees for all the molecular graphs in the dataset.

        Returns:
            root_node_tensor: tensor (batch_size x hidden_size)
                The root encoding vectors for all the junction trees.
        """

        # obtain the word embeddings for all the cluster nodes in the dataset
        word_embeddings = []

        for junc_tree in junc_tree_batch:
            for node in junc_tree.nodes:
                word_embedding = self.embedding(node.wid)
                word_embeddings.append(word_embedding)


        # convert list of tensors to 2D tensor
        word_embeddings = torch.stack(word_embedding, dim=0)

        # obtain the list of all root nodes in the dataset
        root_batch = [junc_tree.nodes[0] for junc_tree in junc_tree_batch]

        # obtain the bottom-up traversal order for all the junction-trees in the dataset
        traversal_order_list = []

        for root in root_batch:
            traversal_order = self.get_bottom_up_traversal_order(root)
            traversal_order_list.append(traversal_order)

        # implement convolution
        prev_layer_output = word_embeddings

        for convnet_layer in range(self.convnet_layers):
            layer_output = convnet_layer(root_batch, prev_layer_output, traversal_order_list)
            prev_layer_output = layer_output

        # retrieve the idxs of all the root nodes in the dataset
        root_node_idxs = [root.idx for root in root_batch]

        # retrieve the hidden feature vectors for all the root nodes in the dataset
        root_node_tensor = torch.index_select(source=layer_output, dim=0, index=torch.tensor(root_node_idxs))

        return root_node_tensor

    def get_bottom_up_traversal_order(self, root):
        """
        This method, gets the bottom-up traversal order for implementing the graph convolution while traversing
        the junction-tree from bottom-up.

        * node.idx is the id of the node across all nodes, of all junction trees, for all molecules of the dataset.

        Args:
        root: Root of junction tree of a molecule in the training dataset.

        Returns:
            traversal_order: List of lists of tuples. Each sublist of tuples corresponds to a depth of junction tree.
                            Each tuple corresponds to an edge along which message passing occurs.
        """

        # FIFO queue for BFS traversal
        fifo_queue = deque([root])

        # set to keep track of visited nodes
        visited = set([root.idx])

        # root node is at zeroth depth
        root.depth = 0

        # list to store appropriate traversal order
        bottom_up = []

        while len(fifo_queue) > 0:
            # pop node from front of the queue
            x = fifo_queue.popleft()

            # traverse the neighbors
            for y in x.neighbors:
                if y.idx not in visited:
                    fifo_queue.append(y)

                    visited.add(y.idx)

                    y.depth = x.depth + 1

                    if y.depth > len(bottom_up):
                        # have a separate sublist for every depth
                        bottom_up.append([])

                    bottom_up[y.depth - 1].append((y, x))

        # first we implement bottom-up traversal and then top-down traversal
        traversal_order = bottom_up[::-1]

        return traversal_order




# class JTNNEncoder(nn.Module):
#     """
#     Junction Tree Neural Network Encoder
#     """
#
#     def __init__(self, vocab, hidden_size, embedding=None):
#         """
#         The constructor for the class.
#
#         Args:
#             vocab: The cluster vocabulary over the entire training dataset.
#             hidden_size: Dimension of the encoding space.
#             embedding: Embedding space for encoding vocabulary composition.
#         """
#         # invoke the superclass constructor
#         super(JTNNEncoder, self).__init__()
#
#         # size of hidden "edge message vectors"
#         self.hidden_size = hidden_size
#
#         # size of the vocabulary of clusters
#         self.vocab_size = vocab.size()
#
#         # the entire vocabulary of clusters
#         self.vocab = vocab
#
#         if embedding is None:
#             self.embedding = nn.Embedding(self.vocab_size, hidden_size)
#         else:
#             self.embedding = embedding
#
#         # all the weight matrices for the GRU
#         self.W_z = nn.Linear(2 * hidden_size, hidden_size)
#         self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.U_r = nn.Linear(hidden_size, hidden_size)
#         self.W_h = nn.Linear(2 * hidden_size, hidden_size)
#         self.W = nn.Linear(2 * hidden_size, hidden_size)
#
#     def forward(self, root_batch):
#         """
#         Args:
#             root_batch: list / batch of root nodes of the corresponding junction trees of the batch of molecules.
#
#         Returns:
#             h: Dictionary containing hidden message vectors for all the edges, of all the junction trees, across the
#                entire training dataset.
#             root_vecs; Root vectors for all the junction trees, across the entire training dataset.
#         """
#         # list to store bottom-up & top-down traversal order for each junction tree
#
#         traversal_order_list = []
#
#         # get the traversal order for each junction tree given root node
#         for root in root_batch:
#             traversal_order = self.get_bottom_up_top_down_traversal_order(root)
#             traversal_order_list.append(traversal_order)
#
#         # dictionary for storing hidden messages along various edges
#         h = {}
#
#         max_depth = max([len(traversal_order) for traversal_order in traversal_order_list])
#
#         # if no messages from any neighbor node, then use this vector of zeros as
#         # neighbor message vector
#         padding = create_var(torch.zeros(self.hidden_size), False)
#
#         for timestep in range(max_depth):
#
#             edge_tuple_list = []
#
#             for traversal_order in traversal_order_list:
#                 # keep appending traversal orders for a particular depth level,
#                 # from a given traversal_order list,
#                 # until the list is not empty
#                 if timestep < len(traversal_order):
#                     edge_tuple_list.extend(traversal_order[timestep])
#
#             # for each edge, list of wids (word_idx corresponding to the cluster vocabulary item) of the starting node.
#             cur_x = []
#
#             # hidden messages for the current timestep, for the junction trees, across the entire training dataset.
#             cur_h_nei = []
#
#             for node_x, node_y in edge_tuple_list:
#                 x, y = node_x.idx, node_y.idx
#                 # wid is the index of the SMILES string, corresponding to the vocabulary cluster
#                 # of the node
#                 cur_x.append(node_x.wid)
#
#                 # hidden messages from predecessor neighbor nodes of x, to x
#                 h_nei = []
#
#                 for node_z in node_x.neighbors:
#                     z = node_z.idx
#                     if z == y:
#                         continue
#                     # hidden message from predecessor neighbor node z to node x
#                     h_nei.append(h[(z, x)])
#
#                 # each node can have at most MAX_NUM_NEIGHBORS(= 8) neighbors
#                 # thus we have a fixed construct of 8 message vectors
#                 # if a node doesn't receive messages from all of 8 neighbors,
#                 # then we set these message vectors to the zero vector
#                 pad_len = MAX_NUM_NEIGHBORS - len(h_nei)
#                 h_nei.extend([padding] * pad_len)
#
#                 # append the chunk of hidden message vectors from neighbors to the cur_h_nei list
#                 # for batch operation
#                 cur_h_nei.extend(h_nei)
#
#             # for each wid in the list, get the corresponding word embedding
#             cur_x = create_var(torch.LongTensor(cur_x))
#             cur_x = self.embedding(cur_x)
#
#             # hidden edge message vector for this timestep, for all the junction trees, across the entire
#             # training dataset.
#             cur_h_nei = torch.cat(cur_h_nei, dim=0).view(-1, MAX_NUM_NEIGHBORS, self.hidden_size)
#
#             # calculate the hidden messages for the next timestep, using the GRU operation.
#             new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
#
#             # put the hidden messages for the next timestep, in the dictionary.
#             for idx, edge in enumerate(edge_tuple_list):
#                 x, y = edge[0].idx, edge[1].idx
#                 h[(x, y)] = new_h[idx]
#
#         # evaluate root vectors encoding the structure for all the junction tress, across the entire training dataset.
#         root_vecs = self.evaluate_root_vecs(root_batch, h, self.embedding, self.W)
#
#         return h, root_vecs
#
#     def get_bottom_up_top_down_traversal_order(self, root):
#         """
#         This method, gets the bottom-up and top-down traversal order for tree message passing purposes.
#
#         * node.idx is the id of the node across all nodes, of all junction trees, for all molecules of the dataset.
#
#         Args:
#         root: Root of junction tree of a molecule in the training dataset.
#
#         Returns:
#             traversal_order: List of lists of tuples. Each sublist of tuples corresponds to a depth of junction tree.
#                             Each tuple corresponds to an edge along which message passing occurs.
#         """
#
#         # FIFO queue for BFS traversal
#         fifo_queue = deque([root])
#
#         # set to keep track of visited nodes
#         visited = set([root.idx])
#
#         # root node is at zeroth depth
#         root.depth = 0
#
#         # list to store appropriate traversal order
#         top_down, bottom_up = [], []
#
#         while len(fifo_queue) > 0:
#             # pop node from front of the queue
#             x = fifo_queue.popleft()
#
#             # traverse the neighbors
#             for y in x.neighbors:
#                 if y.idx not in visited:
#                     fifo_queue.append(y)
#
#                     visited.add(y.idx)
#
#                     y.depth = x.depth + 1
#
#                     if y.depth > len(top_down):
#                         # have a separate sublist for every depth
#                         top_down.append([])
#                         bottom_up.append([])
#
#                     top_down[y.depth - 1].append((x, y))
#                     bottom_up[y.depth - 1].append((y, x))
#
#         # first we implement bottom-up traversal and then top-down traversal
#         traversal_order = bottom_up[::-1] + top_down
#
#         return traversal_order
#
#     def evaluate_root_vecs(self, root_batch, h, embedding, W):
#         """
#         This method, returns the hidden vectors for the root nodes for all the junction trees, across the entire
#         training dataset.
#
#         Args:
#             root_batch: list / batch of root nodes of the corresponding junction trees of the batch of molecules
#             h: dictionary of hidden messages along all the edges, of all the junction trees, across the entire training dataset.
#             embedding: embedding space for vocabulary composition
#             W: weight matrix for calculating the hidden vectors for the root nodes of the junction trees, across the entire
#             training dataset.
#
#         Returns:
#             root_vecs: Hidden vectors for the root nodes of all the junction trees, across all the molecules of the
#             training dataset.
#         """
#
#         # for each root node, store the idx of the corresponding cluster vocabulary item.
#         x_idx = []
#
#         # list to store lists of hidden edge message vectors from neighbors to root, for all root
#         # nodes in the root_batch
#         h_nei = []
#
#         hidden_size = embedding.embedding_dim
#
#         padding = create_var(torch.zeros(hidden_size), False)
#
#         for root in root_batch:
#             x_idx.append(root.wid)
#
#             # list to store hidden edge messages from neighbors of each root node
#             hidden_edge_messages = [h[(node_y.idx, root.idx)] for node_y in root.neighbors]
#
#             # each node can have at most MAX_NUM_NEIGHBORS(= 8 ) neighbors
#             # thus we have a fixed construct of 8 message vectors
#             # if a node doesn't receive messages from all of 8 neighbors,
#             # then we set these message vectors to the zero vector
#             pad_len = MAX_NUM_NEIGHBORS - len(hidden_edge_messages)
#             hidden_edge_messages.extend([padding] * pad_len)
#             h_nei.extend(hidden_edge_messages)
#
#         h_nei = torch.cat(h_nei, dim=0).view(-1, MAX_NUM_NEIGHBORS, hidden_size)
#
#         sum_h_nei = h_nei.sum(dim=1)
#
#         x_vec = create_var(torch.LongTensor(x_idx))
#
#         x_vec = embedding(x_vec)
#
#         root_vecs = torch.cat([x_vec, sum_h_nei], dim=1)
#         return nn.ReLU()(W(root_vecs))

