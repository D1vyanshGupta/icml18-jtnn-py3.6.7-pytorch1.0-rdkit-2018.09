import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from MolJuncTreeNode import MolJuncTreeNode
from MolJuncTreeNode import MolJuncTreeNode
from ClusterVocab import ClusterVocab

from nnutils import create_var, GRU
from chemutils import enum_assemble

MAX_NUM_NEIGHBORS = 15
MAX_DECODE_LEN = 100


class JTNNDecoder(nn.Module):

    """
    Description: This module implements the Tree Decoder given the latent represenation of the root node of a junction-tree. (Section 2.4)
    """
    def __init__(self, vocab, hidden_size, latent_size, vocab_embedding=None):
        """
        Description: The constructor for the class.

        Args:
            vocab: List[MolJuncTreeNode]
                The list of cluster vocabulary over the entire training dataset.

            hidden_size: int
                The dimension of the vector encoding space.

            latent_size: int
                The dimension of the latent space.

            vocab_embedding:
                The embedding space for obtaining embedding vectors given cluster vocabulary item's idx.
        """

        # invoke superclass constructor
        super(JTNNDecoder, self).__init__()

        # size of the hidden edge message vector
        self.hidden_size = hidden_size

        # number of vocabulary clusters
        self.vocab_size = vocab.size()

        # the entire vocabulary of clusters
        self.vocab = vocab

        self.vocab_embedding = nn.Embedding(self.vocab_size, hidden_size)

        # GRU weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        # label prediction weights
        self.W = nn.Linear(hidden_size + latent_size, hidden_size)

        # topological prediction weights
        self.U = nn.Linear(hidden_size + latent_size, hidden_size)
        self.U_i = nn.Linear(2 * hidden_size, hidden_size)

        # output weights
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_o = nn.Linear(hidden_size, 1)

        # loss functions
        self.pred_loss = nn.CrossEntropyLoss(size_average=False)
        self.stop_loss = nn.BCEWithLogitsLoss(size_average=False)

    def aggregate(self, hiddens, contexts, tree_vecs, mode):
        """
        Args:
            hiddens: torch.tensor (shape: batch_size x hidden_size)
                The hidden message vectors for the corresponding tree vectors.

            contexts: torch.LongTensor (shape: # x hidden_size)
                The tree vector representations for those junction-trees, that are to be used in the current computation.

            tree_vecs: torch.tensor (shape: batch_size x hidden_size)
                The tree vector representations for all the junction-trees, across the entire dataset.

            mode: str
                Whether topological or label prediction is being done.

        Returns:
        """
        if mode == 'word':
            V, V_o = self.W, self.W_o
        elif mode == 'stop':
            V, V_o = self.U, self.U_o
        else:
            raise ValueError('attention mode is wrong')

        tree_contexts = tree_vecs.index_select(0, contexts)

        input_vec = torch.cat([hiddens, tree_contexts], dim=-1)

        output_vec = F.relu( V(input_vec) )

        return V_o(output_vec)

    # def get_trace(self, node):
    #     # create a virtual node, because root node has no parent node
    #     virtual_node = MolJuncTreeNode("")
    #     virtual_node.idx = -1
    #
    #     # stack to store dfs traversal order
    #     trace = []
    #     self.dfs(trace, node, virtual_node)
    #     return [(x.smiles, y.smiles, direction) for x, y, direction in trace]

    def forward(self, junc_tree_batch, tree_vecs):
        """
        Args:
            junc_tree_batch: List[MolJuncTree]
                The list of junction-trees for all the molecules, across the entire dataset.

            tree_vecs: torch.tensor (shape: batch_size x hidden_size)
                The vector represenations of all the junction-trees, for all the molecules, across the entire dataset.
        """
        # initialize
        pred_hiddens, pred_contexts, pred_targets = [], [], []
        stop_hiddens, stop_contexts, stop_targets = [], [], []

        # list to store dfs traversals for molecular trees of all molecules
        traces = []

        for junc_tree in junc_tree_batch:
            stack = []
            # root node has no parent node,
            # so we use a virtual node with idx = -1
            self.dfs(stack, junc_tree.nodes[0], -1)
            traces.append(stack)

            for node in junc_tree.nodes:
                node.neighbors = []


        # predict root
        batch_size = len(junc_tree_batch)
        pred_hiddens.append(create_var(torch.zeros(batch_size, self.hidden_size)))

        # list of indices of cluster vocabulary items,
        # for the root node of all junction trees, across the entire dataset.
        pred_targets.extend([junc_tree.nodes[0].wid for junc_tree in junc_tree_batch])

        pred_contexts.append(create_var(torch.LongTensor(range(batch_size))))

        # number of traversals to go through, to ensure that dfs traversal is completed for the
        # junction-tree with the largest size / height.
        max_iter = max([len(tr) for tr in traces])

        # padding vector for putting in place of messages from non-existant neighbors
        padding = create_var(torch.zeros(self.hidden_size), False)

        # dictionary to store hidden edge message vectors
        h = {}

        for iter in range(max_iter):
            # list to store edge tuples that will be considered in this iteration.
            edge_tuple_list = []

            # batch id of all junc_trees / tree_vecs whose edge_tuple
            # in being considered in this timestep
            batch_list = []

            for idx, dfs_traversal in enumerate(traces):
                # keep appending traversal orders for a particular depth level,
                # from a given traversal_order list,
                # until the list is not empty
                if iter < len(dfs_traversal):
                    edge_tuple_list.append(dfs_traversal[iter])
                    batch_list.append(idx)

            cur_x = []
            cur_h_nei, cur_o_nei = [], []

            for node_x, real_y, _ in edge_tuple_list:
                # neighbors for message passing (target not included)
                # hidden edge message vectors from predecessor neighbor nodes
                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
                # a node can at max MAX_NUM_NEIGHBORS(=15) neighbors
                # if it has less neighbors, then we append vector of zeros as messages from non-existent neighbors
                pad_len = MAX_NUM_NEIGHBORS - len(cur_nei)
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                # neighbors for stop (topological) prediction (all neighbors)
                # hidden edge messages from all neighbor nodes
                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
                pad_len = MAX_NUM_NEIGHBORS - len(cur_nei)
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len)

                # current cluster embedding
                cur_x.append(node_x.wid)

            # cluster embedding
            cur_x = create_var(torch.LongTensor(cur_x))
            cur_x = self.vocab_embedding(cur_x)

            # implement message passing
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1, MAX_NUM_NEIGHBORS, self.hidden_size)
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            # node aggregate
            cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NUM_NEIGHBORS, self.hidden_size)
            cur_o = cur_o_nei.sum(dim=1)

            # gather targets
            pred_target, pred_list = [], []
            stop_target = []

            # teacher forcing
            for idx, edge_tuple in enumerate(edge_tuple_list):
                node_x, node_y, direction = edge_tuple
                x, y = node_x.idx, node_y.idx
                h[(x, y)] = new_h[idx]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(idx)
                stop_target.append(direction)

            # hidden states for stop (topological) prediction
            cur_batch = create_var(torch.LongTensor(batch_list))
            stop_hidden = torch.cat([cur_x, cur_o], dim=1)
            stop_hiddens.append(stop_hidden)
            stop_contexts.append(cur_batch)
            stop_targets.extend(stop_target)

            # hidden states for cluster prediction
            if len(pred_list) > 0:
                batch_list = [batch_list[idx] for idx in pred_list]
                cur_batch = create_var(torch.LongTensor(batch_list))
                pred_contexts.append(cur_batch)

                cur_pred = create_var(torch.LongTensor(pred_list))
                pred_hiddens.append(new_h.index_select(0, cur_pred))
                pred_targets.extend(pred_target)

        # last stop at root
        cur_x, cur_o_nei = [], []
        for junc_tree in junc_tree_batch:
            node_x = junc_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NUM_NEIGHBORS - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = create_var(torch.LongTensor(cur_x))
        cur_x = self.vocab_embedding(cur_x)
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NUM_NEIGHBORS, self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x, cur_o], dim=1)
        stop_hiddens.append(stop_hidden)
        stop_contexts.append(create_var(torch.LongTensor(range(batch_size))))
        stop_targets.extend([0] * len(junc_tree_batch))

        # predict next cluster
        pred_contexts = torch.cat(pred_contexts, dim=0)
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_scores = self.aggregate(pred_hiddens, pred_contexts, tree_vecs, 'word')
        pred_targets = create_var(torch.LongTensor(pred_targets))

        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(junc_tree_batch)
        _, preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        # predict stop
        stop_contexts = torch.cat(stop_contexts, dim=0)
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_hiddens = F.relu( self.U_i(stop_hiddens) )
        stop_scores = self.aggregate(stop_hiddens, stop_contexts, tree_vecs, 'stop')
        stop_scores = stop_scores.squeeze(-1)
        stop_targets = create_var(torch.Tensor(stop_targets))

        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(junc_tree_batch)
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()

    def decode(self, tree_vecs):
        """
        Description: Given the tree vector, predict the corresponding junction-tree.

        Args:
            tree_vecs: torch.tensor (shape: hidden_size)

        Returns:
            root: MolJuncTreeNode
                The root node of the decoded junction-tree.

            all_nodes: List[MolJuncTreeNode]
                The list of all the nodes in the decoded junction-tree.
        """
        assert tree_vecs.size(0) == 1

        stack = []
        init_hiddens = create_var( torch.zeros(1, self.hidden_size) )
        zero_pad = create_var(torch.zeros(1,1,self.hidden_size))
        contexts = create_var( torch.LongTensor(1).zero_() )

        # root Prediction
        root_score = self.aggregate(init_hiddens, contexts, tree_vecs, 'word')
        _, root_wid = torch.max(root_score, dim=1)
        root_wid = root_wid.item()

        root = MolJuncTreeNode(self.vocab.get_smiles(root_wid))
        root.wid = root_wid
        root.idx = 0
        stack.append((root, self.vocab.get_slots(root.wid)))

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x, parent_slot = stack[-1]
            cur_h_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1, -1, self.hidden_size)
            else:
                cur_h_nei = zero_pad

            cur_x = create_var(torch.LongTensor([node_x.wid]))
            cur_x = self.vocab_embedding(cur_x)

            # Predict stop
            cur_h = cur_h_nei.sum(dim=1)
            stop_hidden = torch.cat([cur_x, cur_h], dim=1)
            stop_hiddens = F.relu(self.U(stop_hidden))
            stop_score = self.aggregate(stop_hiddens, contexts, tree_vecs, 'stop')

            backtrack = (stop_score.item() < 0)

            # go down forward: predict next cluster
            if not backtrack:
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                pred_score = self.aggregate(new_h, contexts, tree_vecs, 'word')

                _,sort_wid = torch.sort(pred_score, dim=1, descending=True)
                sort_wid = sort_wid.data.squeeze()

                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = MolJuncTreeNode(self.vocab.get_smiles(wid))
                    if self.have_slots(parent_slot, slots) and self.can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                # no more children can be added
                if next_wid is None:
                    backtrack = True
                else:
                    node_y = MolJuncTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.wid = next_wid
                    node_y.idx = len(all_nodes)
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx,node_y.idx)] = new_h[0]
                    stack.append( (node_y,next_slots) )
                    all_nodes.append(node_y)

            # backtrack, use if instead of else
            if backtrack:
                if len(stack) == 1:
                    # back to root, terminate
                    break

                parent_node,_ = stack[-2]
                cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != parent_node.idx ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                h[(node_x.idx,parent_node.idx)] = new_h[0]
                parent_node.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes

    # do dfs traversal,
    # (r, a, 1) signifies that dfs traversal along r --> a has started
    # (a, r, 0) signifies that dfs traversal along a --> r has terminated
    def dfs(self, stack, x, parent_idx):
        for y in x.neighbors:
            # don't do anything for parent node
            if y.idx == parent_idx:
                continue
            stack.append((x, y, 1))
            self.dfs(stack, y, x.idx)
            stack.append((y, x, 0))

    # chemistry domain knowledge
    def have_slots(self, parent_slots, child_slots):
        if len(parent_slots) > 2 and len(child_slots) > 2:
            return True
        matches = []
        for idx_i, slot_1 in enumerate(parent_slots):
            atom_1, charge_1, num_hydrogen_1 = slot_1
            for idx_j, slot_2 in enumerate(child_slots):
                atom_2, charge_2, num_hydrogen_2 = slot_2
                if atom_1 == atom_2 and charge_1 == charge_2 and (atom_1 != "C" or num_hydrogen_1 + num_hydrogen_2 >= 4):
                    matches.append((idx_i, idx_j))

        if len(matches) == 0:
            return False

        parent_match, child_match = zip(*matches)
        if len(set(parent_match)) == 1 and 1 < len(parent_slots) <= 2:  # never remove atom from ring
            parent_slots.pop(parent_match[0])
        if len(set(child_match)) == 1 and 1 < len(child_slots) <= 2:  # never remove atom from ring
            child_slots.pop(child_match[0])

        return True

    def can_assemble(self, node_x, node_y):
        neighbors = node_x.neighbors + [node_y]
        for idx, neighbor_node in enumerate(neighbors):
            neighbor_node.nid = idx

        # exclude nodes corresponding to "singleton-clusters"
        neighbors = [neighbor_node for neighbor_node in neighbors if neighbor_node.mol.GetNumAtoms() > 1]

        # sort neighbor nodes in descending order of number of atoms
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)

        # obtain neighbor nodes corresponding to "singleton-clusters"
        singletons = [neighbor_node for neighbor_node in neighbors if neighbor_node.mol.GetNumAtoms() == 1]

        neighbors = singletons + neighbors

        # retrieve all possible candidates molecular attachment configurations of node_x with its neighbor nodes
        candidates = enum_assemble(node_x, neighbors)
        return len(candidates) > 0
