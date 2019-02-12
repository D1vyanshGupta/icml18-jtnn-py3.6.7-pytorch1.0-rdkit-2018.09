import torch
import torch.nn as nn

from nnutils import create_var
from JTNNEncoder import JTNNEncoder
from JTNNDecoder import JTNNDecoder
from MessPassNet import MessPassNet
from JTMessPassNet import JTMessPassNet
from MolGraphEncoder import MolGraphEncoder
from MolJuncTree import MolJuncTree

from chemutils import enum_assemble, set_atom_map, deep_copy_mol, attach_mols
from new_datautils import set_batch_nodeID

import rdkit.Chem as Chem
import copy


class JTNNVAE(nn.Module):
    """
    Description: This class is the implementation of the Junction Tree Variational Auto Encoder.
    """

    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG, num_layers, use_graph_conv, training_prop_pred=True):
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
        """

        # invoke superclass constructor
        super(JTNNVAE, self).__init__()

        # whether to use message passing or graph convnet
        self.use_graph_conv = use_graph_conv

        self.training_prop_pred = training_prop_pred

        # cluster vocabulary over the entire dataset.
        self.vocab = vocab

        # size of hidden layer
        self.hidden_size = hidden_size

        # size of latent space (latent representation involves both tree and graph encoding vectors)
        self.latent_size = latent_size = latent_size // 2

        if self.use_graph_conv:
            # for encoding molecular graphs, to hidden vector representation
            self.graph_enc = MolGraphEncoder(hidden_size, num_layers)

        else:
            # encoder for producing the molecule graph encoding given batch of molecules
            self.mpn = MessPassNet(hidden_size, depthG)

            # for encoding candidate subgraphs, in the graph decoding phase (section 2.5)
            self.jtmpn = JTMessPassNet(hidden_size, depthG)

        self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size))

        self.decoder = JTNNDecoder(vocab, hidden_size, latent_size, nn.Embedding(vocab.size(), hidden_size))

        # weight matrices for calculating mean and log_var vectors, for implementing the VAE
        self.T_mean = nn.Linear(hidden_size, latent_size)

        self.T_var = nn.Linear(hidden_size, latent_size)

        self.G_mean = nn.Linear(hidden_size, latent_size)

        self.G_var = nn.Linear(hidden_size, latent_size)

        self.A_assm = nn.Linear(latent_size, hidden_size, bias=False)

        # reconstruction loss
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.propNN = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
        self.prop_loss = nn.MSELoss()

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
        z_log_var = -torch.abs(W_var(z_vecs))  # Following Mueller et al.

        # as per Kingma & Welling
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size

        # reparameterization trick
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon

        return z_vecs, kl_loss

    def sample_prior(self):
        z_tree = torch.randn(1, self.latent_size).cuda()
        z_mol = torch.randn(1, self.latent_size).cuda()
        return self.decode(z_tree, z_mol)

    def sample_prior_graph_conv(self):
        z_tree = torch.randn(1, self.latent_size).cuda()
        z_mol = torch.randn(1, self.latent_size).cuda()
        return self.decode_graph_conv(z_tree, z_mol)

    def encode_latent_mean(self, jtenc_holder, mpn_holder):
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        return torch.cat([tree_mean, mol_mean], dim=1)

    def encode_latent_mean_graph_conv(self, jtenc_holder, molenc_holder):
        tree_vecs, mol_vecs = self.encode_graph_conv(jtenc_holder, molenc_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        return torch.cat([tree_mean, mol_mean], dim=1)

    def forward(self, x_batch, beta, _lambda):
        if self.use_graph_conv:
            x_junc_tree_batch, x_jtenc_holder, x_molenc_holder, x_cand_molenc_holder, prop_batch = x_batch
            x_tree_vecs, x_mol_vecs = self.encode_graph_conv(x_jtenc_holder, x_molenc_holder)
            z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
            z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

            kl_div = tree_kl + mol_kl
            word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_junc_tree_batch, z_tree_vecs)
            assm_loss, assm_acc = self.assm_graph_conv(x_junc_tree_batch, x_cand_molenc_holder, z_mol_vecs)

            if self.training_prop_pred:
                all_vec = torch.cat([z_tree_vecs, z_mol_vecs], dim=1)
                prop_label = create_var(torch.Tensor(prop_batch))
                prop_loss = self.prop_loss(self.propNN(all_vec).squeeze(), prop_label)
                return word_loss + topo_loss + assm_loss + beta * kl_div + prop_loss, kl_div.item(), word_acc, topo_acc, assm_acc, prop_loss

            else:
                all_vec = torch.cat([z_tree_vecs, z_mol_vecs], dim=1)
                prop_val = self.propNN(all_vec)
                prop_mean = prop_val.mean()
                return _lambda * (word_loss + topo_loss + assm_loss + beta * kl_div) + (1 - _lambda) * prop_mean, kl_div.item(), word_acc, topo_acc, assm_acc, prop_mean

        else:
            x_junc_tree_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder, prop_batch = x_batch
            x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
            z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
            z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

            kl_div = tree_kl + mol_kl
            word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_junc_tree_batch, z_tree_vecs)
            assm_loss, assm_acc = self.assm(x_junc_tree_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess)

            if self.training_prop_pred:
                all_vec = torch.cat([z_tree_vecs, z_mol_vecs], dim=1)
                prop_label = create_var(torch.Tensor(prop_batch))
                prop_loss = self.prop_loss(self.propNN(all_vec).squeeze(), prop_label)
                return word_loss + topo_loss + assm_loss + beta * kl_div + prop_loss, kl_div.item(), word_acc, topo_acc, assm_acc, prop_loss

            else:
                all_vec = torch.cat([z_tree_vecs, z_mol_vecs], dim=1)
                prop_label = create_var(torch.Tensor(prop_batch))
                prop_val = self.propNN(all_vec)
                prop_mean = prop_val.mean()
                return _lambda * (word_loss + topo_loss + assm_loss + beta * kl_div) + \
                       (1 - _lambda) * prop_mean, kl_div.item(), word_acc, topo_acc, assm_acc, prop_mean

            return word_loss + topo_loss + assm_loss + beta * kl_div, kl_div.item(), word_acc, topo_acc, assm_acc

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

        candidate_vecs = self.jtmpn(atom_feature_matrix, bond_feature_matrix, atom_adjacency_graph,
                                    bond_adjacency_graph, scope, x_tree_mess)

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

    def decode(self, x_tree_vecs, x_mol_vecs):
        # currently do not support batch decoding
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
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None)
        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atom_map(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None

    def decode_graph_conv(self, x_tree_vecs, x_mol_vecs):
        # currently do not support batch decoding
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

        # bilinear
        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze()

        cur_mol = deep_copy_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol = self.dfs_assemble_graph_conv(x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None)
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

        cand_smiles, cand_mols, cand_amap = zip(*cands)
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

    def dfs_assemble_graph_conv(self, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node):
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

        cand_smiles, cand_mols, cand_amap = zip(*cands)

        jt_graph_enc_holder = MolGraphEncoder.tensorize(cand_smiles)
        cand_vecs = self.graph_enc(*jt_graph_enc_holder)

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
                cur_mol = self.dfs_assemble_graph_conv(x_mol_vecs, all_nodes, cur_mol, new_global_amap, pred_amap,
                                                       nei_node, cur_node)
                if cur_mol is None:
                    result = False
                    break
            if result:
                return cur_mol

    def reconstruct(self, smiles):
        junc_tree = MolJuncTree(smiles)
        junc_tree.recover()

        set_batch_nodeID([junc_tree], self.vocab)

        jtenc_holder, _ = JTNNEncoder.tensorize([junc_tree])
        mpn_holder = MessPassNet.tensorize([smiles])

        tree_vec, _, mol_vec = self.encode(jtenc_holder, mpn_holder)

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))  # Following Mueller et al.

        epsilon = create_var(torch.randn(1, self.latent_size), False)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = create_var(torch.randn(1, self.latent_size), False)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon

        return self.decode(tree_vec, mol_vec)

    def reconstruct_graph_conv(self, smiles):
        junc_tree = MolJuncTree(smiles)
        junc_tree.recover()

        set_batch_nodeID([junc_tree], self.vocab)

        jtenc_holder, _ = JTNNEncoder.tensorize([junc_tree])
        molenc_holder = MolGraphEncoder.tensorize([smiles])

        tree_vec, mol_vec = self.encode_graph_conv(jtenc_holder, molenc_holder)

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))  # Following Mueller et al.

        epsilon = create_var(torch.randn(1, self.latent_size), False)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = create_var(torch.randn(1, self.latent_size), False)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
        return self.decode_graph_conv(tree_vec, mol_vec)
