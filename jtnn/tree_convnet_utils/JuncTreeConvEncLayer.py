import torch
import torch.nn as nn

from nnutils import create_var

MAX_NUM_NEIGHBORS = 8


class ConvEncLayer(nn.Module):
    """
    Description: This class implements a single graph convolutional layer for encoding junction trees.
    """

    def __init__(self, hidden_size):
        """
        Description: Constructor for the class.

        Args:
            hidden_size: int
                The dimension of the hidden message vectors.
        """

        # invoke superclass constructor
        super(ConvEncLayer, self).__init__()

        # the dimension of the hidden vectors
        self.hidden_size = hidden_size

        # weight matrix for hidden vector of current node
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)

        # weight matrix for Hadamard product of the edge gates and child hidden vectors
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)

        # weight matrix for hidden vector of current node (for edge gate calculation)
        self.A = nn.Linear(hidden_size, hidden_size, bias=False)

        # weight matrix for hidden vector of child nodes (for edge gate calculation)
        self.b = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, root_batch, prev_layer_output, traversal_order_list):

        # number of nodes in the entire dataset
        num_nodes = prev_layer_output.shape[0]

        # placeholder to store the result from convolution of this layer
        current_layer_output = torch.zeros(num_nodes, self.hidden_size)

        # use this as hidden message vector from non-existent neighbor / child node
        padding = create_var(torch.zeros(self.hidden_size), False)

        # maximum number of iterations for which to run the bottom-up tree convolution
        max_iter = max([len(tr) for tr in traversal_order_list])

        # apply graph convolution for all the non-root nodes
        for iter in range(max_iter):

            # list to store edges, for this iteration of bottom-up convolution, across various junction-trees,
            # for the entire dataset
            edge_tuple_list = []

            for traversal_order in traversal_order_list:
                # keep appending traversal orders for a particular depth level,
                # from a given traversal_order list,
                # until the list is not empty
                if iter < len(traversal_order):
                    edge_tuple_list.extend(traversal_order[iter])

            # list to store hidden vectors of child nodes, of all the current nodes, for this iteration
            iter_child_hidden_vecs = []

            # list to store idxs of parent nodes for this iteration
            parent_node_idxs = []

            for node_x, node_y in edge_tuple_list:
                # retrieve the node idxs of node_x and node_y
                x, y = node_x.idx, node_y.idx

                # node_x is the parent node, for whom, the hidden vectors of child nodes are being aggregated
                parent_node_idxs.append(x)

                # retrieve the idx of child nodes, of node_x
                child_node_idxs = [node_z.idx for node_z in node_x.neighbors if node_z.idx != y]

                # list to store hidden vectors of child nodes
                child_hidden_vecs = []

                if len(child_node_idxs) > 0:
                    child_hidden_vecs = list(torch.index_select(source=prev_layer_output, dim=0, index=torch.tensor(child_node_idxs)))

                # (Chemistry Domain-Knowledge):
                # each node can have at most MAX_NUM_NEIGHBORS(= 8) neighbors in any junction-tree.
                # thus for each node, we expect 8 hidden vectors from child / neighbor nodes
                # if a node has less than 8 child nodes,
                # then we consider the messages from non-existent child nodes to be the zero vector of appropriate size.
                pad_len = MAX_NUM_NEIGHBORS - len(child_hidden_vecs)
                child_hidden_vecs.extend([padding] * pad_len)

                # append the chunk of hidden vectors of child nodes, of all the parent nodes, for this iteration,
                # to iter_child_hidden_vecs, for batch operation
                iter_child_hidden_vecs.extend(child_hidden_vecs)

            # convert the list of tensors, to a 2D tensor
            iter_child_hidden_tensor = torch.stack(iter_child_hidden_vecs, dim=0)

            # retrieve the hidden vectors for the parent nodes, for this iteration
            iter_parent_hidden_tensor = torch.index_select(source=prev_layer_output, dim=0, index=torch.tensor(parent_node_idxs))

            # compute the edge gates for child nodes, of all the parent nodes, for this iteration
            iter_edge_gates = self.compute_edge_gates(iter_parent_hidden_tensor, iter_child_hidden_tensor)

            # compute the new hidden vectors of the parent nodes, for this iteration, by applying the graph convolution
            iter_new_hidden_tesor = self.apply_graph_convolution(iter_parent_hidden_tensor, iter_child_hidden_tensor, iter_edge_gates)

            # assign the new hidden vectors to the appropriate parent nodes
            current_layer_output[parent_node_idxs] = iter_new_hidden_tesor

        # apply graph convolution for all the root nodes

        # list to store idxs of root nodes
        root_node_idxs = []

        # list to store hidden vectors of child nodes, for all root nodes
        root_child_hidden_vecs = []

        for root in root_batch:
            # root node, for whom, the hidden vectors of all its child nodes are being aggregated
            root_node_idxs.append(root.idx)

            # list to store idxs of child nodes, of this root node
            child_node_idxs = [node_y.idx for node_y in root.neighbors]

            # every root node has at least 1 child node

            # retrieve the hidden vectors of the root node
            child_hidden_vecs = torch.index_select(source=prev_layer_output, dim=0, index=torch.tensor(child_node_idxs))

            pad_len = MAX_NUM_NEIGHBORS - len(child_hidden_vecs)
            child_hidden_vecs.extend([padding] * pad_len)

            root_child_hidden_vecs.extend(child_hidden_vecs)

        # convert the list of tensors, to a 2D tensor
        root_child_hidden_tensor = torch.stack(root_child_hidden_vecs, dim=0)

        # retrieve the hidden vectors, for the root nodes
        root_hidden_tensor = torch.index_select(source=prev_layer_output, dim=0, index=torch.tensor(parent_node_idxs))

        # compute the edge gates for child nodes, of all the root nodes
        root_child_edge_gates = self.compute_edge_gates(root_hidden_tensor, root_child_hidden_tensor)

        # compute the new hidden vectors of the root nodes, by applying the graph convolution
        new_root_hidden_tensor = self.apply_graph_convolution(root_hidden_tensor, root_child_hidden_tensor, root_child_edge_gates)

        # assign the new hidden vectors to the appropriate root nodes
        current_layer_output[root_node_idxs] = new_root_hidden_tensor

        return current_layer_output

    def apply_graph_convolution(self, iter_parent_hidden_tensor, iter_child_hidden_tensor, iter_edge_gates):
        # multiply the parent node hidden vectors with the U matrix
        U_h = self.U(iter_parent_hidden_tensor)

        # multiply the child node hidden vectors with the V matrix
        V_h = self.V(iter_child_hidden_tensor)

        # resize the V_h tensor for taking the Hadamard product with the edge gates
        V_h= V_h.view(-1, MAX_NUM_NEIGHBORS, self.hidden_size)

        # evaluate the Hadamard product
        edge_gate_mul_hidden_vecs = iter_edge_gates + V_h

        # sum up all the Hadamard products, corresponding to every parent node, for this iteration
        edge_gate_mul_hidden_vecs_sum = torch.sum(edge_gate_mul_hidden_vecs, dim=1)

        # evaluate the synaptic input to the convolutional layer
        synaptic_input = U_h + edge_gate_mul_hidden_vecs_sum

        # apply the ReLU activation
        new_hidden_vecs = nn.ReLU()(synaptic_input)

        return new_hidden_vecs

    def compute_edge_gates(self, iter_parent_hidden_tensor, iter_child_hidden_tensor):
        # multiply the child node hidden vectors with the B matrix
        B_h = self.B(iter_child_hidden_tensor)

        # aggregate all the hidden tensors, as per parent nodes
        B_h = B_h.view(-1, MAX_NUM_NEIGHBORS, self.hidden_size)

        # multiply the parent node hidden vectors with the A matrix
        A_h = self.A(iter_parent_hidden_tensor)

        # add a new dimension to A_h, for broadcasting purposes
        A_h = A_h.unsqueeze(1)

        # calculate the sum for edge-gate computation
        eta = A_h + B_h

        # apply the sigmoid activation
        eta = nn.Sigmoid()(eta)

        return eta
