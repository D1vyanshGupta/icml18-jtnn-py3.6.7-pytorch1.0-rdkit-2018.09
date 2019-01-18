import torch
import torch.nn as nn

from nnutils import create_var
from JTNNEncoder import JTNNEncoder
from JTNNDecoder import JTNNDecoder
from MessPassNet import MessPassNet
from JTMessPassNet import JTMessPassNet
from MolGraphEncoder import MolGraphEncoder

from chemutils import enum_assemble, set_atom_map, deep_copy_mol, attach_mols, decode_stereo

import rdkit.Chem as Chem
import copy


class JTNNVAE(nn.Module):
    """
    Description: This class is the implementation of the Junction Tree Variational Auto Encoder.
    """

    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG, num_layers, use_graph_conv, share_embedding=False):
        """
        Description: This is the constructor for the class.

        Args:
            vocab: List[MolJuncTreeNode]
                The cluster vocabulary over the dataset.

            hidden_size: int
                The dimension of the embedding space.

            latent_size: int
                The dimension of the latent space.

            depthT: int
                The number of timesteps for implementing message passing for encoding the junction trees.

            depthG: int
                The number of timesteps for implementing message passing for encoding the molecular graphs.

            num_layers: int
                The number of layers for the graph convolutional encoder.

            use_graph_conv: Boolean
                Whether to use the Graph ConvNet or Message Passing for encoding molecular graphs.

            share_embedding: Boolean
                Whether to share the embedding space between
        """

        # invoke superclass constructor
        super(JTNNVAE, self).__init__()

        # whether to use message passing or graph convnet
        self.use_graph_conv = use_graph_conv

        # cluster vocabulary over the entire dataset.
        self.vocab = vocab

        # size of hidden layer 
        self.hidden_size = hidden_size

        # size of latent space (latent representation involves both tree and graph encoding vectors)
        self.latent_size = latent_size = latent_size // 2

        if share_embedding:
            self.embedding = nn.Embedding(vocab.size(), hidden_size)
            self.jtnn = JTNNEncoder(hidden_size, depthT, self.embedding)
            self.decoder = JTNNDecoder(vocab, hidden_size, latent_size, self.embedding)
        else:
            self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size))
            self.decoder = JTNNDecoder(vocab, hidden_size, latent_size, nn.Embedding(vocab.size(), hidden_size))

        if self.use_graph_conv is True:
            # for encoding molecular graphs, to hidden vector representation
            self.graph_enc = MolGraphEncoder(hidden_size, num_layers)
        else:
            # encoder for producing the molecule graph encoding given batch of molecules
            self.mpn = MessPassNet(hidden_size, depthG)

            # for encoding candidate subgraphs, in the graph decoding phase (section 2.5)
            self.jtmpn = JTMessPassNet(hidden_size, depthG)

        # weight matrices for calculating mean and log_var vectors, for implementing the VAE
        self.T_mean = nn.Linear(hidden_size, latent_size)

        self.T_var = nn.Linear(hidden_size, latent_size)

        self.G_mean = nn.Linear(hidden_size, latent_size)

        self.G_var = nn.Linear(hidden_size, latent_size)

        #
        self.A_assm = nn.Linear(latent_size, hidden_size, bias=False)

        # reconstruction loss
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)
        self.stereo_loss = nn.CrossEntropyLoss(size_average=False)

    # def set_batch_node_id(self, junc_tree_batch, vocab):
    #     """
    #     This method, given the junction trees for all the molecules in the training dataset,
    #     gives a unique idx to every node of all the junction trees.
    #
    #     This is done to implement tree message passing, in a very strange albeit workable manner.
    #
    #     Args:
    #         junc_tree_batch: List of all the junction trees of all the molecules in the training dataset.
    #         vocab: The cluster vocabulary over the entire training dataset
    #     """
    #
    #     tot = 0
    #     for junc_tree in junc_tree_batch:
    #         for node in junc_tree.nodes:
    #             # set a unique idx for this node across all nodes, of all junction trees, in the training dataset.
    #             node.idx = tot
    #
    #             # word idx.
    #             # for each node, set its wid to the idx of the corresponding cluster vocabulary item.
    #             node.wid = vocab.get_index(node.smiles)
    #             tot += 1

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        return tree_vecs, tree_mess, mol_vecs

    def encode_graph_conv(self, jtenc_holder, molenc_holder):
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        mol_vecs = self.graph_enc(*molenc_holder)
        return tree_vecs, mol_vecs

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.

        # as per Kingma & Welling
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size

        # reparameterization trick
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon

        return z_vecs, kl_loss

    def sample_prior(self):
        z_tree = torch.randn(1, self.latent_size).cuda()
        z_mol = torch.randn(1, self.latent_size).cuda()
        # z_tree = torch.randn(1, self.latent_size)
        # z_mol = torch.randn(1, self.latent_size)
        return self.decode(z_tree, z_mol)

    # def encode(self, junc_tree_batch):
    #     """
    #     This method, given the junction trees for all the molecules in the dataset,
    #     produces the encoding for the junction trees and the molecular graphs of all the molecules.
    #
    #     Args:
    #         junc_tree_batch: List of all the junction trees of all the molecules in the training dataset.
    #
    #     Returns:
    #         tree_mess: Dictionary containing tree messages, for all the junction trees, in the training dataset.
    #         tree_vec: Encoding vectors for all the junction trees, in the training dataset.
    #         mol_vec: Encoding vectors for all the molecular graphs, in the training dataset.
    #     """
    #
    #     # for each node, across all molecules in the dataset, give each node an id
    #     self.set_batch_node_id(junc_tree_batch, self.vocab)
    #
    #     # list of all root nodes for the junction trees of all molecules in the dataset
    #     root_batch = [mol_tree.nodes[0] for mol_tree in junc_tree_batch]
    #
    #     # tree_message dictionary, and tree_vectors for all molecules in the dataset
    #     tree_mess, tree_vecs = self.jtnn(root_batch)
    #
    #     # SMILES representation of all the molecules in the dataset
    #     smiles_batch = [mol_tree.smiles for mol_tree in junc_tree_batch]
    #
    #     # graph encoding vector for all the molecules in the dataset
    #     if self.use_graph_conv:
    #         mol_vecs = self.graph_enc(smiles_batch)
    #         return tree_vecs, mol_vecs
    #
    #     else:
    #         mol_vecs = self.mpn(smiles_batch)
    #         return tree_mess, tree_vecs, mol_vecs

    # def encode_latent_mean(self, smiles_list):
    #     mol_batch = [MolJuncTree(s) for s in smiles_list]
    #     for mol_tree in mol_batch:
    #         mol_tree.recover()
    #
    #     _, tree_vec, mol_vec = self.encode(mol_batch)
    #     tree_mean = self.T_mean(tree_vec)
    #     mol_mean = self.G_mean(mol_vec)
    #     return torch.cat([tree_mean, mol_mean], dim=1)

    def forward(self, x_batch, beta):
        if self.use_graph_conv:
            x_junc_tree_batch, x_jtenc_holder, x_molenc_holder, x_cand_molenc_holder, x_stereo_molenc_holder = x_batch
            x_tree_vecs, x_mol_vecs = self.encode_graph_conv(x_jtenc_holder, x_molenc_holder)
            z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
            z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

            kl_div = tree_kl + mol_kl
            word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_junc_tree_batch, z_tree_vecs)
            assm_loss, assm_acc = self.assm_graph_conv(x_junc_tree_batch, x_cand_molenc_holder, z_mol_vecs)
            stereo_loss, stereo_acc = self.stereo_graph_conv(x_stereo_molenc_holder, z_mol_vecs)

            return word_loss + topo_loss + assm_loss + stereo_loss + beta * kl_div, kl_div.item(), word_acc, topo_acc, assm_acc, stereo_acc
        else:
            x_junc_tree_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder, x_stereo_molenc_holder = x_batch
            x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
            z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
            z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

            kl_div = tree_kl + mol_kl
            word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_junc_tree_batch, z_tree_vecs)
            assm_loss, assm_acc = self.assm(x_junc_tree_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess)
            stereo_loss, stereo_acc = self.stereo(x_stereo_molenc_holder, z_mol_vecs)

            return word_loss + topo_loss + assm_loss + stereo_loss + beta * kl_div, kl_div.item(), word_acc, topo_acc, assm_acc, stereo_acc

    # graph decoder
    def assm_graph_conv(self, junc_tree_batch, x_cand_molenc_holder, z_mol_vecs):
        cand_molenc_holder, batch_idx = x_cand_molenc_holder
        batch_idx = create_var(batch_idx)

        candidate_vecs = self.graph_enc(*cand_molenc_holder)

        z_mol_vecs = z_mol_vecs.index_select(0, batch_idx)
        z_mol_vecs = self.A_assm(z_mol_vecs)  # bilinear
        scores = torch.bmm(
            z_mol_vecs.unsqueeze(1),
            candidate_vecs.unsqueeze(-1)
        ).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(junc_tree_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.candidates) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.candidates.index(node.label)
                num_candidates = len(node.candidates)
                cur_score = scores.narrow(0, tot, num_candidates)
                tot += num_candidates

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        all_loss = sum(all_loss) / len(junc_tree_batch)
        return all_loss, acc * 1.0 / cnt

    def assm(self, junc_tree_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess):
        jtmpn_holder, batch_idx = x_jtmpn_holder
        atom_feature_matrix, bond_feature_matrix, atom_adjacency_graph, bond_adjacency_graph, scope = jtmpn_holder

        batch_idx = create_var(batch_idx)

        candidate_vecs = self.jtmpn(atom_feature_matrix, bond_feature_matrix, atom_adjacency_graph, bond_adjacency_graph, scope, x_tree_mess)

        z_mol_vecs = z_mol_vecs.index_select(0, batch_idx)
        z_mol_vecs = self.A_assm(z_mol_vecs)  # bilinear
        scores = torch.bmm(
            z_mol_vecs.unsqueeze(1),
            candidate_vecs.unsqueeze(-1)
        ).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(junc_tree_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.candidates) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.candidates.index(node.label)
                num_candidates = len(node.candidates)
                cur_score = scores.narrow(0, tot, num_candidates)
                tot += num_candidates

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        all_loss = sum(all_loss) / len(junc_tree_batch)
        return all_loss, acc * 1.0 / cnt

    def stereo_graph_conv(self, x_stereo_molenc_holder, z_mol_vecs):
        stereo_molenc_holder, stereo_batch_idx, labels = x_stereo_molenc_holder

        if len(labels) == 0:
            return create_var(torch.zeros(1)), 1.0

        batch_idx = create_var(torch.LongTensor(stereo_batch_idx))
        stereo_candidates_vecs = self.graph_enc(*stereo_molenc_holder)
        stereo_labels = z_mol_vecs.index_select(0, batch_idx)
        stereo_labels = self.A_assm(stereo_labels)
        scores = torch.nn.CosineSimilarity()(stereo_candidates_vecs, stereo_labels)

        st,acc = 0,0
        all_loss = []
        for label,le in labels:
            cur_scores = scores.narrow(0, st, le)
            if cur_scores.data[label].item() >= cur_scores.max().item():
                acc += 1
            label = create_var(torch.LongTensor([label]))
            all_loss.append( self.stereo_loss(cur_scores.view(1,-1), label) )
            st += le
        #all_loss = torch.cat(all_loss).sum() / len(labels)
        all_loss = sum(all_loss) / len(stereo_labels)
        return all_loss, acc * 1.0 / len(stereo_labels)

    def stereo(self, x_stereo_molenc_holder, z_mol_vecs):
        stereo_molenc_holder, stereo_batch_idx, labels = x_stereo_molenc_holder

        if len(labels) == 0:
            return create_var(torch.zeros(1)), 1.0

        batch_idx = create_var(torch.LongTensor(stereo_batch_idx))
        stereo_candidates_vecs = self.mpn(*stereo_molenc_holder)
        stereo_labels = z_mol_vecs.index_select(0, batch_idx)
        stereo_labels = self.A_assm(stereo_labels)
        scores = torch.nn.CosineSimilarity()(stereo_candidates_vecs, stereo_labels)

        st, acc = 0, 0
        all_loss = []

        for label, le in labels:
            cur_scores = scores.narrow(0, st, le)
            if cur_scores.data[label].item() >= cur_scores.max().item():
                acc += 1
            label = create_var(torch.LongTensor([label]))
            all_loss.append( self.stereo_loss(cur_scores.view(1,-1), label) )
            st += le
        #all_loss = torch.cat(all_loss).sum() / len(labels)
        all_loss = sum(all_loss) / len(stereo_labels)
        return all_loss, acc * 1.0 / len(stereo_labels)

    def decode(self, x_tree_vecs, x_mol_vecs):
        #currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root, pred_nodes = self.decoder.decode(x_tree_vecs)
        if len(pred_nodes) == 0:
            return None
        elif len(pred_nodes) == 1:
            return pred_root.smiles

        # mark nid & is_leaf & atommap
        for idx, node in enumerate(pred_nodes):
            node.nid = idx + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atom_map(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _, tree_mess = self.jtnn(*jtenc_holder)

        # important: tree_mess is a matrix, mess_dict is a python dict
        tree_mess = (tree_mess, mess_dict)

        # bilinear
        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze()

        cur_mol = deep_copy_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None)
        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atom_map(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None

    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0:
            return None

        cand_smiles, cand_amap = zip(*cands)
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        jtmpn_holder = JTMessPassNet.tensorize(cands, y_tree_mess[1])
        fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])

        scores = torch.mv(cand_vecs, x_mol_vecs)
        _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)

        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)  # father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            result = True
            for nei_node in children:
                if nei_node.is_leaf:
                    continue
                cur_mol = self.dfs_assemble(y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap, pred_amap,
                                            nei_node, cur_node)
                if cur_mol is None:
                    result = False
                    break
            if result:
                return cur_mol

    # def forward(self, junc_tree_batch, beta=0):
    #     batch_size = len(junc_tree_batch)
    #
    #     # tree_message dictionary,
    #     # junction tree encoding vectors and
    #     # molecular graph encoding vectors for all molecules in the dataset
    #
    #     # junc_tree_batch = self.convert_tensor_batch_to_junc_tree_batch(tensor_batch)
    #
    #     if self.use_graph_conv:
    #         tree_vecs, mol_vecs = self.encode(junc_tree_batch)
    #
    #     else:
    #         tree_mess, tree_vecs, mol_vecs = self.encode(junc_tree_batch)
    #
    #     tree_mean = self.T_mean(tree_vecs)
    #     tree_log_var = -torch.abs(self.T_var(tree_vecs))  # Following Mueller et al.
    #
    #     mol_mean = self.G_mean(mol_vecs)
    #     mol_log_var = -torch.abs(self.G_var(mol_vecs))  # Following Mueller et al.
    #
    #     z_mean = torch.cat([tree_mean, mol_mean], dim=1)
    #     z_log_var = torch.cat([tree_log_var, mol_log_var], dim=1)
    #
    #     # calculate KL divergence between q(z|x) and p(z)
    #     kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
    #
    #     # reparameterization trick
    #     epsilon = create_var(torch.randn(batch_size, self.latent_size // 2), False)
    #     tree_vecs = tree_mean + torch.exp(tree_log_var // 2) * epsilon
    #
    #     # reparameterization trick
    #     epsilon = create_var(torch.randn(batch_size, self.latent_size // 2), False)
    #     mol_vecs = mol_mean + torch.exp(mol_log_var // 2) * epsilon
    #
    #     label_pred_loss, topo_loss, label_pred_acc, topo_acc = self.decoder(junc_tree_batch, tree_vecs)
    #
    #     if self.use_graph_conv:
    #         assm_loss, assm_acc = self.assm_new(junc_tree_batch, mol_vecs)
    #
    #     else:
    #         assm_loss, assm_acc = self.assm_use_graph_conv(junc_tree_batch, mol_vecs, tree_mess)
    #
    #     stereo_loss, stereo_acc = self.stereo(junc_tree_batch, mol_vecs)
    #     all_vec = torch.cat([tree_vecs, mol_vecs], dim=1)
    #     loss = label_pred_loss + topo_loss + assm_loss + 2 * stereo_loss + beta * kl_loss
    #
    #     # return loss, kl_loss.item(), label_pred_acc, topo_acc, assm_acc, stereo_acc
    #     return loss, kl_loss.item(), label_pred_loss, topo_loss, assm_loss, stereo_loss
    #     # return loss

    # graph decoding loss
    # def assm_use_graph_conv(self, junc_tree_batch, mol_vecs, tree_mess):
    #     candidates = []
    #     batch_idx = []
    #
    #     for idx, junc_tree in enumerate(junc_tree_batch):
    #         for node in junc_tree.nodes:
    #             # leaf node's attachment is determined by neighboring node's attachment
    #             if node.is_leaf or len(node.candidates) == 1:
    #                 continue
    #
    #             candidates.extend([(candidate_mol, junc_tree.nodes, node) for candidate_mol in node.candidate_mols])
    #             batch_idx.extend([idx] * len(node.candidates))
    #
    #     # encode all candidate assembly configurations, for all nodes, of all
    #     # junction trees, across the entire dataset.
    #     candidate_vecs = self.jtmpn(candidates, tree_mess)
    #     candidate_vecs = self.G_mean(candidate_vecs)
    #
    #     batch_idx = create_var(torch.LongTensor(batch_idx))
    #     mol_vecs = mol_vecs.index_select(0, batch_idx)
    #
    #     mol_vecs = mol_vecs.view(-1, 1, self.latent_size // 2)
    #     candidate_vecs = candidate_vecs.view(-1, self.latent_size // 2, 1)
    #     scores = torch.bmm(mol_vecs, candidate_vecs).squeeze()
    #
    #     count, tot, acc = 0, 0, 0
    #     all_loss = []
    #     for idx, junc_tree in enumerate(junc_tree_batch):
    #         comp_nodes = [node for node in junc_tree.nodes if len(node.candidates) > 1 and not node.is_leaf]
    #         count += len(comp_nodes)
    #         for node in comp_nodes:
    #             label_idx = node.candidates.index(node.label)
    #             num_candidates = len(node.candidates)
    #             cur_score = scores.narrow(0, tot, num_candidates)
    #             tot += num_candidates
    #
    #             if cur_score.data[label_idx] >= cur_score.max().item():
    #                 acc += 1
    #
    #             label_idx = create_var(torch.LongTensor([label_idx]))
    #             all_loss.append(self.assm_loss(cur_score.view(1, -1), label_idx))
    #
    #     # all_loss = torch.stack(all_loss).sum() / len(mol_batch)
    #     all_loss = sum(all_loss) / len(junc_tree_batch)
    #     return all_loss, acc * 1.0 / count

    # def assm_new(self, junc_tree_batch, mol_vecs):
    #     candidates = []
    #     batch_idx = []
    #
    #     for idx, junc_tree in enumerate(junc_tree_batch):
    #         for node in junc_tree.nodes:
    #             # leaf node's attachment is determined by neighboring node's attachment
    #             if node.is_leaf or len(node.candidates) == 1:
    #                 continue
    #
    #             candidates.extend([candidate_smiles for candidate_smiles in node.candidates])
    #
    #             batch_idx.extend([idx] * len(node.candidates))
    #
    #     # encode all candidate assembly configurations, for all nodes, of all
    #     # junction trees, across the entire dataset.
    #
    #     candidate_vecs = self.graph_enc(candidates)
    #
    #     candidate_vecs = self.G_mean(candidate_vecs)
    #
    #     batch_idx = create_var(torch.LongTensor(batch_idx))
    #     mol_vecs = mol_vecs.index_select(0, batch_idx)
    #     #
    #     mol_vecs = mol_vecs.view(-1, 1, self.latent_size // 2)
    #     candidate_vecs = candidate_vecs.view(-1, self.latent_size // 2, 1)
    #     scores = torch.bmm(mol_vecs, candidate_vecs).squeeze()
    #     #
    #     count, tot, acc = 0, 0, 0
    #     all_loss = []
    #     for idx, junc_tree in enumerate(junc_tree_batch):
    #         comp_nodes = [node for node in junc_tree.nodes if len(node.candidates) > 1 and not node.is_leaf]
    #         count += len(comp_nodes)
    #         for node in comp_nodes:
    #             label_idx = node.candidates.index(node.label)
    #             num_candidates = len(node.candidates)
    #             cur_score = scores.narrow(0, tot, num_candidates)
    #             tot += num_candidates
    #
    #             if cur_score.data[label_idx] >= cur_score.max().item():
    #                 acc += 1
    #
    #             label_idx = create_var(torch.LongTensor([label_idx]))
    #             all_loss.append(self.assm_loss(cur_score.view(1, -1), label_idx))
    #
    #     # all_loss = torch.stack(all_loss).sum() / len(mol_batch)
    #     all_loss = sum(all_loss) / len(junc_tree_batch)
    #     return all_loss, acc * 1.0 / count

    # stereo loss
    # def stereo(self, junc_tree_batch, mol_vecs):
    #     stereo_candidates, batch_idx = [], []
    #     labels = []
    #     for idx, junc_tree in enumerate(junc_tree_batch):
    #         candidates = junc_tree.stereo_candidates
    #         if len(candidates) == 1:
    #             continue
    #         if junc_tree.smiles3D not in candidates:
    #             candidates.append(junc_tree.smiles3D)
    #         stereo_candidates.extend(candidates)
    #         batch_idx.extend([idx] * len(candidates))
    #         labels.append((candidates.index(junc_tree.smiles3D), len(candidates)))
    #
    #     if len(labels) == 0:
    #         return create_var(torch.zeros(1)), 1.0
    #
    #     batch_idx = create_var(torch.LongTensor(batch_idx))
    #
    #     if self.use_graph_conv:
    #         stereo_candidates_vecs = self.graph_enc(stereo_candidates)
    #
    #     else:
    #         stereo_candidates_vecs = self.mpn(stereo_candidates)
    #
    #     stereo_candidates_mean_vecs = self.G_mean(stereo_candidates_vecs)
    #     stereo_labels = torch.index_select(input=mol_vecs, dim=0, index=batch_idx)
    #     scores = torch.nn.CosineSimilarity()(stereo_candidates_mean_vecs, stereo_labels)
    #
    #     st, acc = 0, 0
    #     all_loss = []
    #     for label, le in labels:
    #         cur_scores = scores.narrow(0, st, le)
    #         if cur_scores.data[label] >= cur_scores.max().item():
    #             acc += 1
    #         label = create_var(torch.LongTensor([label]))
    #         all_loss.append(self.stereo_loss(cur_scores.view(1, -1), label))
    #         st += le
    #     # all_loss = torch.cat(all_loss).sum() / len(labels)
    #     all_loss = sum(all_loss) / len(labels)
    #     return all_loss, acc * 1.0 / len(labels)

    # def reconstruct(self, smiles, prob_decode=False):
    #     junc_tree = MolJuncTree(smiles)
    #     junc_tree.recover()
    #     _, tree_vec, mol_vec = self.encode([junc_tree])
    #
    #     tree_mean = self.T_mean(tree_vec)
    #     tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
    #     mol_mean = self.G_mean(mol_vec)
    #     mol_log_var = -torch.abs(self.G_var(mol_vec))  # Following Mueller et al.
    #
    #     epsilon = create_var(torch.randn(1, self.latent_size // 2), False)
    #     tree_vec = tree_mean + torch.exp(tree_log_var // 2) * epsilon
    #     epsilon = create_var(torch.randn(1, self.latent_size // 2), False)
    #     mol_vec = mol_mean + torch.exp(mol_log_var // 2) * epsilon
    #     return self.decode(tree_vec, mol_vec, prob_decode)

    # def recon_eval(self, smiles):
    #     junc_tree = MolJuncTree(smiles)
    #     junc_tree.recover()
    #     _, tree_vec, mol_vec = self.encode([junc_tree])
    #
    #     tree_mean = self.T_mean(tree_vec)
    #     tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
    #     mol_mean = self.G_mean(mol_vec)
    #     mol_log_var = -torch.abs(self.G_var(mol_vec))  # Following Mueller et al.
    #
    #     all_smiles = []
    #     for i in range(10):
    #         epsilon = create_var(torch.randn(1, self.latent_size // 2), False)
    #         tree_vec = tree_mean + torch.exp(tree_log_var // 2) * epsilon
    #         epsilon = create_var(torch.randn(1, self.latent_size // 2), False)
    #         mol_vec = mol_mean + torch.exp(mol_log_var // 2) * epsilon
    #         for j in range(10):
    #             new_smiles = self.decode(tree_vec, mol_vec, prob_decode=True)
    #             all_smiles.append(new_smiles)
    #     return all_smiles

    # def sample_prior(self, prob_decode=False):
    #     tree_vec = create_var(torch.randn(1, self.latent_size // 2), False)
    #     mol_vec = create_var(torch.randn(1, self.latent_size // 2), False)
    #     return self.decode(tree_vec, mol_vec, prob_decode)

    # def sample_eval(self):
    #     tree_vec = create_var(torch.randn(1, self.latent_size // 2), False)
    #     mol_vec = create_var(torch.randn(1, self.latent_size // 2), False)
    #     all_smiles = []
    #     for i in range(100):
    #         s = self.decode(tree_vec, mol_vec, prob_decode=True)
    #         all_smiles.append(s)
    #     return all_smiles

    # def decode(self, tree_vec, mol_vec, prob_decode):
    #
    #     pred_root, pred_nodes = self.decoder.decode(tree_vec, prob_decode)
    #
    #     # Mark nid & is_leaf & atommap
    #     for idx, node in enumerate(pred_nodes):
    #         node.nid = idx + 1
    #         node.is_leaf = (len(node.neighbors) == 1)
    #         if len(node.neighbors) > 1:
    #                 set_atom_map(node.mol, node.nid)
    #
    #     tree_mess = self.jtnn([pred_root])[0]
    #
    #     cur_mol = deep_copy_mol(pred_root.mol)
    #     global_atom_map = [{}] + [{} for node in pred_nodes]
    #     global_atom_map[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}
    #
    #     cur_mol = self.dfs_assemble(tree_mess, mol_vec, pred_nodes, cur_mol, global_atom_map, [], pred_root, None,
    #                                 prob_decode)
    #     if cur_mol is None:
    #         return None
    #
    #     cur_mol = cur_mol.GetMol()
    #     set_atom_map(cur_mol)
    #     cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
    #     if cur_mol is None:
    #         return None
    #
    #     smiles2D = Chem.MolToSmiles(cur_mol)
    #     stereo_candidates = decode_stereo(smiles2D)
    #     if len(stereo_candidates) == 1:
    #         return stereo_candidates[0]
    #     stereo_vecs = self.mpn(stereo_candidates)
    #     stereo_vecs = self.G_mean(stereo_vecs)
    #     scores = nn.CosineSimilarity()(stereo_vecs, mol_vec)
    #     _, max_id = scores.max(dim=0)
    #     return stereo_candidates[max_id.item()]

    # def dfs_assemble(self, tree_mess, mol_vec, all_nodes, cur_mol, global_atom_map, parent_atom_map, cur_node, parent_node, prob_decode):
    #     parent_nid = parent_node.nid if parent_node is not None else -1
    #     prev_nodes = [parent_node] if parent_node is not None else []
    #
    #     children = [neighbor_node for neighbor_node in cur_node.neighbors if neighbor_node.nid != parent_nid]
    #
    #     # exclude neighbor nodes corresponding to "singleton clusters"
    #     neighbors = [neighbor_node for neighbor_node in children if neighbor_node.mol.GetNumAtoms() > 1]
    #
    #     # sort neighbor nodes in descending order by number of atoms
    #     neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
    #
    #     # obtain neighbor nodes corresponding to "singleton-clusters"
    #     singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
    #     neighbors = singletons + neighbors
    #
    #     # neighbor_id, ctr_atom_idx, neighbor_atom_idx
    #     cur_atom_map = [(parent_nid, a2, a1) for nid, a1, a2 in parent_atom_map if nid == cur_node.nid]
    #     candidates = enum_assemble(cur_node, neighbors, prev_nodes, cur_atom_map)
    #     if len(candidates) == 0:
    #         return None
    #
    #     candidate_smiles, candidate_mols, candidate_atom_maps = zip(*candidates)
    #
    #     candidates = [(candidate_mol, all_nodes, cur_node) for candidate_mol in candidate_mols]
    #
    #     candidate_vecs = self.jtmpn(candidates, tree_mess)
    #     candidate_vecs = self.G_mean(candidate_vecs)
    #     mol_vec = mol_vec.squeeze()
    #     scores = torch.mv(candidate_vecs, mol_vec) * 20
    #
    #     if prob_decode:
    #         probs = nn.Softmax()(scores.view(1, -1)).squeeze() + 1e-5  # prevent prob = 0
    #         # quick fix
    #         if len(probs.shape) == 0:
    #             probs = torch.tensor([probs])
    #         cand_idx = torch.multinomial(probs, probs.numel())
    #     else:
    #         _, cand_idx = torch.sort(scores, descending=True)
    #
    #     backup_mol = Chem.RWMol(cur_mol)
    #     for idx in range(cand_idx.numel()):
    #         cur_mol = Chem.RWMol(backup_mol)
    #         pred_atom_map = candidate_atom_maps[cand_idx[idx].item()]
    #         new_global_atom_map = copy.deepcopy(global_atom_map)
    #
    #         for neighbor_id, ctr_atom_idx, neighbor_atom_idx in pred_atom_map:
    #             if neighbor_id == parent_nid:
    #                 continue
    #             new_global_atom_map[neighbor_id][neighbor_atom_idx] = new_global_atom_map[cur_node.nid][ctr_atom_idx]
    #
    #         cur_mol = attach_mols(cur_mol, children, [], new_global_atom_map)  # parent is already attached
    #         new_mol = cur_mol.GetMol()
    #         new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
    #
    #         if new_mol is None:
    #             continue
    #
    #         result = True
    #         for child_node in children:
    #             if child_node.is_leaf: continue
    #             cur_mol = self.dfs_assemble(tree_mess, mol_vec, all_nodes, cur_mol, new_global_atom_map, pred_atom_map,
    #                                         child_node, cur_node, prob_decode)
    #             if cur_mol is None:
    #                 result = False
    #                 break
    #         if result:
    #             return cur_mol
    #
    #     return None