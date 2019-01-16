import torch
import torch.nn as nn
import torch.nn.functional as F

from nnutils import create_var, index_select_ND

class GraphGRU(nn.Module):
    """
    Description: This module implements the GRU for message aggregation for Tree Encoding purposes. (Section 2.3)
    """
    def __init__(self, input_size, hidden_size, depth):
        """
        Description: The constructor for the class.

        Args:
            input_size: int

            hidden_size: int
                The dimension of the hidden edge message vectors.

            depth: int
                The number of timesteps for which to implement the message passing.
        """
        # invoke the superclass constructor
        super(GraphGRU, self).__init__()

        # the dimension of the "hidden edge message vectors"
        self.hidden_size = hidden_size

        #
        self.input_size = input_size

        # the number fo timesteps for which to run the message passing
        self.depth = depth

        # GRU weight matrices
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h, x, mess_adjacency_graph):
        """
        Args:
            h: torch.LongTensor (shape: num_message_vectors x hidden_size)
                The hidden message vectors for all the edge vectors.

            x: torch.LongTensor (shape: num_message_vectors x hidden_size)
                The embedding vector for initial nodes of all the edges.

            mess_graph: torch.LongTensor (shape: num_message_vectors x MAX_NUM_NEIGHBORS)
                For each edge, the list of idxs of all "inward hidden edge message vectors" for purposes of node feature aggregation.
        Returns:
            h: torch.LongTensor (shape: num_message_vectors x hidden_size)
                The hidden message vectors for all the edge vectors.
        """
        mask = torch.ones(h.size(0), 1)

        # the first hidden message vector is the padding vector i.e. vector of all zeros
        # so we zero it out
        mask[0] = 0

        mask = create_var(mask)

        # implement message passing from timestep 0 to T (self.depth) - 1
        for timestep in range(self.depth):

            # get "inward hidden message vectors" for all the edges
            hidden_neighbor_mess = index_select_ND(h, 0, mess_adjacency_graph)

            # sum the "inward hidden message vectors" for all the edges
            hidden_neighbor_mess_sum = hidden_neighbor_mess.sum(dim=1)

            # concatenate the embedding vector for initial nodes and the sum of hidden message vectors.
            z_input = torch.cat([x, hidden_neighbor_mess_sum], dim=1)

            # implement GRU operations
            z = F.sigmoid(self.W_z(z_input))
            r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
            r_2 = self.U_r(hidden_neighbor_mess)
            r = F.sigmoid(r_1 + r_2)

            gated_h = r * hidden_neighbor_mess
            sum_gated_h = gated_h.sum(dim=1)
            h_input = torch.cat([x, sum_gated_h], dim=1)
            pre_h = F.tanh(self.W_h(h_input))
            h = (1.0 - z) * hidden_neighbor_mess_sum + z * pre_h
            h = h * mask

        return h