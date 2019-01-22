from collections import deque

import torch
import torch.nn as nn

import torch.nn.functional as F

from nnutils import create_var

MAX_NUM_NEIGHBORS = 8

from ConvNetLayer import ConvNetLayer

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

    # first we implement bottom-up traversal
    traversal_order = bottom_up[::-1]

    return traversal_order


class JTGraphEncoder(nn.Module):
    """
    Description: This module implements the Tree Convolutional Network.
    """

    def __init__(self, hidden_size, num_layers, embedding=None):

        super(JTGraphEncoder, self).__init__()

        # the dimension of the hidden feature vectors to be used
        self.hidden_size = hidden_size

        # number of "convolutional layers" in the graph encoder
        self.num_layers = num_layers

        # embedding space for obtaining embeddings of node features vectors.
        self.embedding = embedding

        # list to store "convolutional layers" for the encoder
        conv_layers = []

        # instantiate the convnet layers for the encoder
        for layer_idx in range(num_layers):
            # input node_FEATURE_DIM and edge_FEATURE_DIM, for instantiating the first layer
            if layer_idx == 0:
                # instantiate a convnet layer
                conv_layer = ConvNetLayer(hidden_size)

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

        # weight matrix for feature vector of node 'i' for edge e_(i, j)
        self.V = nn.Linear(hidden_size, hidden_size, bias=True)

        # weight matrix for feature vector of node 'j' for edge e_(i, j)
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, node_wid_list, node_child_adjacency_graph, node_edge_adjacency_graph, edge_node_adjacency_graph, scope, root_scope):
        # list to store embedding vectors for junction-tree nodes
        node_feature_vecs = []
        
        # padding vector for node features
        node_feature_padding = torch.zeros(self.hidden_size)

        node_feature_vecs.append(node_feature_padding)
        
        # obtain embedding vectors for all the junction-tree nodes
        node_embeddings = self.embedding(node_wid_list)
        
        node_feature_vecs.extend(list(node_embeddings))
        
        node_feature_matrix = torch.stack(node_feature_vecs, dim=0)
        
        total_num_edges = edge_node_adjacency_graph.shape[0]
        
        edge_feature_matrix = torch.zeros(total_num_edges, self.hidden_size)
        
        # create PyTorch variables
        node_feature_matrix = create_var(node_feature_matrix)
        edge_feature_matrix = create_var(edge_feature_matrix)
        node_child_adjacency_graph = create_var(node_child_adjacency_graph)
        node_edge_adjacency_graph = create_var(node_edge_adjacency_graph)
        edge_node_adjacency_graph = create_var(edge_node_adjacency_graph)

        # implement convolution
        node_layer_input = node_feature_matrix
        edge_layer_input = edge_feature_matrix

        for conv_layer in self.conv_layers:
            # implement forward pass for this convolutional layer
            node_layer_output, edge_layer_output = conv_layer(node_layer_input, edge_layer_input, node_child_adjacency_graph,
                                                              node_edge_adjacency_graph, edge_node_adjacency_graph)

            # set the input features for the next convolutional layer
            node_layer_input, edge_layer_input = node_layer_output, edge_layer_output

        # for each molecular graph, pool all the edge feature vectors
        tree_vecs = self.pool_edge_features_for_junc_trees(node_layer_output, edge_layer_output, edge_node_adjacency_graph, scope)

        # tree_vecs = node_layer_output[root_scope]
        
        return tree_vecs 
        
    def pool_edge_features_for_junc_trees(self, node_layer_output, edge_layer_output, edge_node_adjacency_graph, scope):
        """
        Args:
            node_layer_output: torch.tensor (shape: batch_size x node_feature_dim)
                The matrix containing feature vectors, for all the nodes, across the entire batch.

            edge_layer_output: torch.tensor (shape: batch_size x edge_feature_dim)
                The matrix containing feature vectors, for all the edges, across the entire batch.

            edge_node_adjacency_graph: torch.tensor (shape: num_edges x 2)

            scope: List[Tuple(int, int)]
                The list to store tuples (total_edges, num_edges), to keep track of all the edge feature vectors,
                belonging to a particular molecule.

        Returns:
             mol_vecs: troch.tensor (shape: batch_size x hidden_size
        """
        # implement edge gate computation
        edge_gate_x = torch.index_select(input=node_layer_output, dim=0, index=edge_node_adjacency_graph[:, 0])
        edge_gate_y = torch.index_select(input=node_layer_output, dim=0, index=edge_node_adjacency_graph[:, 1])

        edge_gate_synaptic_input = self.U(edge_layer_output) + self.V(edge_gate_x) + self.W(edge_gate_y)

        # apply sigmoid activation for computing edge gates
        edge_gates = F.sigmoid(edge_gate_synaptic_input)

        assert(edge_gates.shape == edge_layer_output.shape)

        # multiply edge features with edge gates
        edge_gates_mul_edge_features_tensor = edge_gates * self.A(edge_layer_output)

        # list to store vector representation for various molecules
        mol_vecs = []
        for start_idx, len in scope:
            mol_vec = edge_gates_mul_edge_features_tensor[start_idx : start_idx + len].sum(dim=0)
            mol_vecs.append(mol_vec)

        # stack all all the mol_vecs into a 2-D tensor
        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs


    @staticmethod
    def tensorize(junc_tree_batch):
        # list to store the idxs of root nodes
        root_scope = []

        # list to store the wids of all the nodes, across the entire batch
        node_wid_list = []

        # list to store, the list of idxs of child ndoes, for all the nodes, across the entire batch
        node_child_adjacency_list = []

        node_child_adjacency_list.append([0] * MAX_NUM_NEIGHBORS)

        # list to store, for each node, across the entire batch, the list of idxs of all the edges, in which it is the initial node
        node_edge_adjacency_list = []

        node_edge_adjacency_list.append([0] * MAX_NUM_NEIGHBORS)

        # for each edge, across the entire batch, the idxs of the 2 nodes, of which the edge is composed of.
        edge_node_adjacency_list = []

        edge_node_adjacency_list.append([0] * 2)

        # list to store the bottom-up traversal order for each junction-tree
        traversal_order_list = []

        for junc_tree in junc_tree_batch:
            root = junc_tree.nodes[0]
            traversal_order = get_bottom_up_traversal_order(root)
            traversal_order_list.append(traversal_order)

            root_scope.append(root.idx + 1)

            for node in junc_tree.nodes:
                node_wid = node.wid
                node_wid_list.append(node_wid)

                # append a sublist to store idxs of child ndoes for this node
                node_child_adjacency_list.append([])

                # append a sublist to store idxs of all edges, in which this is the initial node
                node_edge_adjacency_list.append([])

        edge_offset_idx = 1

        edge_start_idx = 1

        scope = []

        for traversal_order in traversal_order_list:

            for edge_tuple_list in traversal_order:

                for node_x, node_y in edge_tuple_list:
                    x_idx, y_idx = node_x.idx + 1, node_y.idx + 1

                    node_child_adjacency_list[y_idx].append(x_idx)

                    # for edge e_(x, y)
                    node_edge_adjacency_list[x_idx].append(edge_offset_idx)

                    component_node_list = [x_idx, y_idx]
                    edge_node_adjacency_list.append(component_node_list)

                    edge_offset_idx += 1

                    # for edge e_(y, x)
                    node_edge_adjacency_list[y_idx].append(edge_offset_idx)

                    component_node_list = [y_idx, x_idx]
                    edge_node_adjacency_list.append(component_node_list)

                    edge_offset_idx += 1


            _len = 2 * len(edge_tuple_list)

            scope.append((edge_start_idx, _len))

            edge_start_idx += _len

        total_num_nodes = len(node_child_adjacency_list)

        total_num_edges = len(edge_node_adjacency_list)

        node_child_adjacency_graph = torch.zeros(total_num_nodes, MAX_NUM_NEIGHBORS).long()

        node_edge_adjacency_graph = torch.zeros(total_num_edges, MAX_NUM_NEIGHBORS).long()

        edge_node_adjacency_graph = torch.zeros(total_num_edges, 2).long()

        # insert values into the node child adjacency graph
        for node_idx in range(total_num_nodes):
            for idx, child_node_idx in enumerate(node_child_adjacency_list[node_idx]):
                node_child_adjacency_graph[node_idx, idx] = child_node_idx

        # insert values into the node edge adjacency graph
        for node_idx in range(total_num_nodes):
            for idx, edge_idx in enumerate(node_edge_adjacency_list[node_idx]):
                node_edge_adjacency_graph[node_idx, idx] = edge_idx

        # insert values into the edge node adjacency graph
        for edge_idx in range(total_num_edges):
            for idx, node_idx in enumerate(edge_node_adjacency_list[edge_idx]):
                edge_node_adjacency_graph[edge_idx, idx] = node_idx

        node_wid_list = torch.LongTensor(node_wid_list)

        return (node_wid_list, node_child_adjacency_graph, node_edge_adjacency_graph, edge_node_adjacency_graph, scope, root_scope)


