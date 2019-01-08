import torch
import torch.nn as nn
from MolJuncTree import MolJuncTree
from nnutils import create_var
from JTNN_Enc import JTNNEncoder
from JTNN_Dec import JTNNDecoder
from MessPassNet import MessPassNet
from JTMessPassNet import JTMessPassNet
from MolGraphEncoder import MolGraphEncoder

from chemutils import enum_assemble, set_atom_map, copy_edit_mol, attach_mols, decode_stereo
import rdkit.Chem as Chem
import copy


class JTNNVAE(nn.Module):
    """
    This class is the implementation of the Junction Tree Variational Auto Encoder.
    """

    def __init__(self, vocab, hidden_size, latent_size, depth, num_layers, use_graph_conv):
        """
        This is the constructor for the class.

        Args:
            vocab: The cluster vocabulary over the training dataset.
            hidden_size: Dimension of the embedding space.
            latent_size: Dimension of the latent space.
            depth: Number of timesteps for implementing message passing for encoding the junction tree and
            the molecular graph.

            num_layers: int
                The number of layers for the graph convolutional encoder.
        """

        # invoke superclass constructor
        super(JTNNVAE, self).__init__()

        self.use_graph_conv = use_graph_conv

        # cluster vocabulary
        self.vocab = vocab

        # size of hidden layer 
        self.hidden_size = hidden_size

        # size of latent space
        self.latent_size = latent_size

        # number of timesteps for which to run the message passing
        self.depth = depth

        # embedding layer for encoding vocabulary composition
        self.vocab_embedding = nn.Embedding(vocab.size(), hidden_size)

        # for encoding junction tree, to hidden vector representation
        self.jtnn = JTNNEncoder(vocab, hidden_size, self.vocab_embedding)

        # for decoding tree vector, back to junction tree
        self.decoder = JTNNDecoder(vocab, hidden_size, latent_size // 2, self.vocab_embedding)

        if self.use_graph_conv is True:
            # for encoding molecular graphs, to hidden vector representation
            self.graph_enc = MolGraphEncoder(hidden_size, num_layers)
        else:
            # for encoding candidate subgraphs, in the graph decoding phase (section 2.5)
            self.jtmpn = JTMessPassNet(hidden_size, depth)

            # encoder for producing the molecule graph encoding given batch of molecules
            self.mpn = MessPassNet(hidden_size, depth)

        # weight matrices for calculating mean and log_var vectors, for implementing the VAE
        self.T_mean = nn.Linear(hidden_size, latent_size // 2)

        self.T_var = nn.Linear(hidden_size, latent_size // 2)

        self.G_mean = nn.Linear(hidden_size, latent_size // 2)

        self.G_var = nn.Linear(hidden_size, latent_size // 2)

        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.stereo_loss = nn.CrossEntropyLoss(size_average=False)

    def set_batch_node_id(self, junc_tree_batch, vocab):
        """
        This method, given the junction trees for all the molecules in the training dataset,
        gives a unique idx to every node of all the junction trees.

        This is done to implement tree message passing, in a very strange albeit workable manner.

        TODO: Refactor this mechanism to something more sensible.

        Args:
            junc_tree_batch: List of all the junction trees of all the molecules in the training dataset.
            vocab: The cluster vocabulary over the entire training dataset
        """

        tot = 0
        for junc_tree in junc_tree_batch:
            for node in junc_tree.nodes:
                # set a unique idx for this node across all nodes, of all junction trees, in the training dataset.
                node.idx = tot

                # word idx.
                # for each node, set its wid to the idx of the corresponding cluster vocabulary item.
                node.wid = vocab.get_index(node.smiles)
                tot += 1

    def encode(self, junc_tree_batch):
        """
        This method, given the junction trees for all the molecules in the dataset,
        produces the encoding for the junction trees and the molecular graphs of all the molecules.

        Args:
            junc_tree_batch: List of all the junction trees of all the molecules in the training dataset.

        Returns:
            tree_mess: Dictionary containing tree messages, for all the junction trees, in the training dataset.
            tree_vec: Encoding vectors for all the junction trees, in the training dataset.
            mol_vec: Encoding vectors for all the molecular graphs, in the training dataset.
        """

        # for each node, across all molecules in the dataset, give each node an id 
        self.set_batch_node_id(junc_tree_batch, self.vocab)

        # list of all root nodes for the junction trees of all molecules in the dataset
        root_batch = [mol_tree.nodes[0] for mol_tree in junc_tree_batch]

        # tree_message dictionary, and tree_vectors for all molecules in the dataset
        tree_mess, tree_vecs = self.jtnn(root_batch)

        # SMILES representation of all the molecules in the dataset
        smiles_batch = [mol_tree.smiles for mol_tree in junc_tree_batch]

        # graph encoding vector for all the molecules in the dataset
        if self.use_graph_conv:
            mol_vecs = self.graph_enc(smiles_batch)
            return tree_vecs, mol_vecs

        else:
            mol_vecs = self.mpn(smiles_batch)
            return tree_mess, tree_vecs, mol_vecs

    # def encode_latent_mean(self, smiles_list):
    #     mol_batch = [MolJuncTree(s) for s in smiles_list]
    #     for mol_tree in mol_batch:
    #         mol_tree.recover()
    #
    #     _, tree_vec, mol_vec = self.encode(mol_batch)
    #     tree_mean = self.T_mean(tree_vec)
    #     mol_mean = self.G_mean(mol_vec)
    #     return torch.cat([tree_mean, mol_mean], dim=1)

    def convert_tensor_batch_to_junc_tree_batch(self, tensor_batch):
        junc_tree_batch = []
        for tensor in tensor_batch:
            t = tensor[tensor != -1]
            smiles = ''.join(list(map(lambda x: chr(x), t)))
            junc_tree = MolJuncTree(smiles)
            junc_tree.recover()
            junc_tree.assemble()
            junc_tree_batch.append(junc_tree)

        for junc_tree in junc_tree_batch:
            for node in junc_tree.nodes:
                if node.label not in node.candidates:
                    node.candidates.append(node.label)
                    node.candidate_mols.append(node.label_mol)

        return junc_tree_batch

    def forward(self, tensor_batch, beta=0):
        batch_size = tensor_batch.shape[0]

        # tree_message dictionary,
        # junction tree encoding vectors and
        # molecular graph encoding vectors for all molecules in the dataset

        junc_tree_batch = self.convert_tensor_batch_to_junc_tree_batch(tensor_batch)

        if self.use_graph_conv:
            tree_vecs, mol_vecs = self.encode(junc_tree_batch)

        else:
            tree_mess, tree_vecs, mol_vecs = self.encode(junc_tree_batch)

        tree_mean = self.T_mean(tree_vecs)
        tree_log_var = -torch.abs(self.T_var(tree_vecs))  # Following Mueller et al.

        mol_mean = self.G_mean(mol_vecs)
        mol_log_var = -torch.abs(self.G_var(mol_vecs))  # Following Mueller et al.

        z_mean = torch.cat([tree_mean, mol_mean], dim=1)
        z_log_var = torch.cat([tree_log_var, mol_log_var], dim=1)

        # calculate KL divergence between q(z|x) and p(z)
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size

        # reparameterization trick
        epsilon = create_var(torch.randn(batch_size, self.latent_size // 2), False)
        tree_vecs = tree_mean + torch.exp(tree_log_var // 2) * epsilon

        # reparameterization trick
        epsilon = create_var(torch.randn(batch_size, self.latent_size // 2), False)
        mol_vecs = mol_mean + torch.exp(mol_log_var // 2) * epsilon

        label_pred_loss, topo_loss, label_pred_acc, topo_acc = self.decoder(junc_tree_batch, tree_vecs)

        if self.use_graph_conv:
            assm_loss, assm_acc = self.assm_new(junc_tree_batch, mol_vecs)

        else:
            assm_loss, assm_acc = self.assm_use_graph_conv(junc_tree_batch, mol_vecs, tree_mess)

        stereo_loss, stereo_acc = self.stereo(junc_tree_batch, mol_vecs)
        all_vec = torch.cat([tree_vecs, mol_vecs], dim=1)
        loss = label_pred_loss + topo_loss + assm_loss + 2 * stereo_loss + beta * kl_loss

        # return loss, kl_loss.item(), label_pred_acc, topo_acc, assm_acc, stereo_acc
        return loss, kl_loss.item(), label_pred_loss, topo_loss, assm_loss, stereo_loss

    # graph decoding loss
    def assm_use_graph_conv(self, junc_tree_batch, mol_vecs, tree_mess):
        candidates = []
        batch_idx = []

        for idx, junc_tree in enumerate(junc_tree_batch):
            for node in junc_tree.nodes:
                # leaf node's attachment is determined by neighboring node's attachment
                if node.is_leaf or len(node.candidates) == 1:
                    continue

                candidates.extend([(candidate_mol, junc_tree.nodes, node) for candidate_mol in node.candidate_mols])
                batch_idx.extend([idx] * len(node.candidates))

        # encode all candidate assembly configurations, for all nodes, of all
        # junction trees, across the entire dataset.
        candidate_vecs = self.jtmpn(candidates, tree_mess)
        candidate_vecs = self.G_mean(candidate_vecs)

        batch_idx = create_var(torch.LongTensor(batch_idx))
        mol_vecs = mol_vecs.index_select(0, batch_idx)

        mol_vecs = mol_vecs.view(-1, 1, self.latent_size // 2)
        candidate_vecs = candidate_vecs.view(-1, self.latent_size // 2, 1)
        scores = torch.bmm(mol_vecs, candidate_vecs).squeeze()

        count, tot, acc = 0, 0, 0
        all_loss = []
        for idx, junc_tree in enumerate(junc_tree_batch):
            comp_nodes = [node for node in junc_tree.nodes if len(node.candidates) > 1 and not node.is_leaf]
            count += len(comp_nodes)
            for node in comp_nodes:
                label_idx = node.candidates.index(node.label)
                num_candidates = len(node.candidates)
                cur_score = scores.narrow(0, tot, num_candidates)
                tot += num_candidates

                if cur_score.data[label_idx] >= cur_score.max().item():
                    acc += 1

                label_idx = create_var(torch.LongTensor([label_idx]))
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label_idx))

        # all_loss = torch.stack(all_loss).sum() / len(mol_batch)
        all_loss = sum(all_loss) / len(junc_tree_batch)
        return all_loss, acc * 1.0 / count

    def assm_new(self, junc_tree_batch, mol_vecs):
        candidates = []
        batch_idx = []

        for idx, junc_tree in enumerate(junc_tree_batch):
            for node in junc_tree.nodes:
                # leaf node's attachment is determined by neighboring node's attachment
                if node.is_leaf or len(node.candidates) == 1:
                    continue

                candidates.extend([candidate_smiles for candidate_smiles in node.candidates])

                batch_idx.extend([idx] * len(node.candidates))

        # encode all candidate assembly configurations, for all nodes, of all
        # junction trees, across the entire dataset.

        candidate_vecs = self.graph_enc(candidates)

        candidate_vecs = self.G_mean(candidate_vecs)

        batch_idx = create_var(torch.LongTensor(batch_idx))
        mol_vecs = mol_vecs.index_select(0, batch_idx)
        #
        mol_vecs = mol_vecs.view(-1, 1, self.latent_size // 2)
        candidate_vecs = candidate_vecs.view(-1, self.latent_size // 2, 1)
        scores = torch.bmm(mol_vecs, candidate_vecs).squeeze()
        #
        count, tot, acc = 0, 0, 0
        all_loss = []
        for idx, junc_tree in enumerate(junc_tree_batch):
            comp_nodes = [node for node in junc_tree.nodes if len(node.candidates) > 1 and not node.is_leaf]
            count += len(comp_nodes)
            for node in comp_nodes:
                label_idx = node.candidates.index(node.label)
                num_candidates = len(node.candidates)
                cur_score = scores.narrow(0, tot, num_candidates)
                tot += num_candidates

                if cur_score.data[label_idx] >= cur_score.max().item():
                    acc += 1

                label_idx = create_var(torch.LongTensor([label_idx]))
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label_idx))

        # all_loss = torch.stack(all_loss).sum() / len(mol_batch)
        all_loss = sum(all_loss) / len(junc_tree_batch)
        return all_loss, acc * 1.0 / count

    # stereo loss
    def stereo(self, junc_tree_batch, mol_vecs):
        stereo_candidates, batch_idx = [], []
        labels = []
        for idx, junc_tree in enumerate(junc_tree_batch):
            candidates = junc_tree.stereo_candidates
            if len(candidates) == 1:
                continue
            if junc_tree.smiles3D not in candidates:
                candidates.append(junc_tree.smiles3D)
            stereo_candidates.extend(candidates)
            batch_idx.extend([idx] * len(candidates))
            labels.append((candidates.index(junc_tree.smiles3D), len(candidates)))

        if len(labels) == 0:
            return create_var(torch.zeros(1)), 1.0

        batch_idx = create_var(torch.LongTensor(batch_idx))

        if self.use_graph_conv:
            stereo_candidates_vecs = self.graph_enc(stereo_candidates)

        else:
            stereo_candidates_vecs = self.mpn(stereo_candidates)

        stereo_candidates_mean_vecs = self.G_mean(stereo_candidates_vecs)
        stereo_labels = torch.index_select(input=mol_vecs, dim=0, index=batch_idx)
        scores = torch.nn.CosineSimilarity()(stereo_candidates_mean_vecs, stereo_labels)

        st, acc = 0, 0
        all_loss = []
        for label, le in labels:
            cur_scores = scores.narrow(0, st, le)
            if cur_scores.data[label] >= cur_scores.max().item():
                acc += 1
            label = create_var(torch.LongTensor([label]))
            all_loss.append(self.stereo_loss(cur_scores.view(1, -1), label))
            st += le
        # all_loss = torch.cat(all_loss).sum() / len(labels)
        all_loss = sum(all_loss) / len(labels)
        return all_loss, acc * 1.0 / len(labels)

    def reconstruct(self, smiles, prob_decode=False):
        junc_tree = MolJuncTree(smiles)
        junc_tree.recover()
        _, tree_vec, mol_vec = self.encode([junc_tree])

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))  # Following Mueller et al.

        epsilon = create_var(torch.randn(1, self.latent_size // 2), False)
        tree_vec = tree_mean + torch.exp(tree_log_var // 2) * epsilon
        epsilon = create_var(torch.randn(1, self.latent_size // 2), False)
        mol_vec = mol_mean + torch.exp(mol_log_var // 2) * epsilon
        return self.decode(tree_vec, mol_vec, prob_decode)

    def recon_eval(self, smiles):
        junc_tree = MolJuncTree(smiles)
        junc_tree.recover()
        _, tree_vec, mol_vec = self.encode([junc_tree])

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))  # Following Mueller et al.

        all_smiles = []
        for i in range(10):
            epsilon = create_var(torch.randn(1, self.latent_size // 2), False)
            tree_vec = tree_mean + torch.exp(tree_log_var // 2) * epsilon
            epsilon = create_var(torch.randn(1, self.latent_size // 2), False)
            mol_vec = mol_mean + torch.exp(mol_log_var // 2) * epsilon
            for j in range(10):
                new_smiles = self.decode(tree_vec, mol_vec, prob_decode=True)
                all_smiles.append(new_smiles)
        return all_smiles

    def sample_prior(self, prob_decode=False):
        tree_vec = create_var(torch.randn(1, self.latent_size // 2), False)
        mol_vec = create_var(torch.randn(1, self.latent_size // 2), False)
        return self.decode(tree_vec, mol_vec, prob_decode)

    def sample_eval(self):
        tree_vec = create_var(torch.randn(1, self.latent_size // 2), False)
        mol_vec = create_var(torch.randn(1, self.latent_size // 2), False)
        all_smiles = []
        for i in range(100):
            s = self.decode(tree_vec, mol_vec, prob_decode=True)
            all_smiles.append(s)
        return all_smiles

    def decode(self, tree_vec, mol_vec, prob_decode):

        pred_root, pred_nodes = self.decoder.decode(tree_vec, prob_decode)

        # Mark nid & is_leaf & atommap
        for idx, node in enumerate(pred_nodes):
            node.nid = idx + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                    set_atom_map(node.mol, node.nid)

        tree_mess = self.jtnn([pred_root])[0]

        cur_mol = copy_edit_mol(pred_root.mol)
        global_atom_map = [{}] + [{} for node in pred_nodes]
        global_atom_map[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol = self.dfs_assemble(tree_mess, mol_vec, pred_nodes, cur_mol, global_atom_map, [], pred_root, None,
                                    prob_decode)
        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atom_map(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        if cur_mol is None:
            return None

        smiles2D = Chem.MolToSmiles(cur_mol)
        stereo_candidates = decode_stereo(smiles2D)
        if len(stereo_candidates) == 1:
            return stereo_candidates[0]
        stereo_vecs = self.mpn(stereo_candidates)
        stereo_vecs = self.G_mean(stereo_vecs)
        scores = nn.CosineSimilarity()(stereo_vecs, mol_vec)
        _, max_id = scores.max(dim=0)
        return stereo_candidates[max_id.item()]

    def dfs_assemble(self, tree_mess, mol_vec, all_nodes, cur_mol, global_atom_map, parent_atom_map, cur_node, parent_node,
                     prob_decode):
        parent_nid = parent_node.nid if parent_node is not None else -1
        prev_nodes = [parent_node] if parent_node is not None else []

        children = [neighbor_node for neighbor_node in cur_node.neighbors if neighbor_node.nid != parent_nid]

        # exclude neighbor nodes corresponding to "singleton clusters"
        neighbors = [neighbor_node for neighbor_node in children if neighbor_node.mol.GetNumAtoms() > 1]

        # sort neighbor nodes in descending order by number of atoms
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)

        # obtain neighbor nodes corresponding to "singleton-clusters"
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        # neighbor_id, ctr_atom_idx, neighbor_atom_idx
        cur_atom_map = [(parent_nid, a2, a1) for nid, a1, a2 in parent_atom_map if nid == cur_node.nid]
        candidates = enum_assemble(cur_node, neighbors, prev_nodes, cur_atom_map)
        if len(candidates) == 0:
            return None

        candidate_smiles, candidate_mols, candidate_atom_maps = zip(*candidates)

        candidates = [(candidate_mol, all_nodes, cur_node) for candidate_mol in candidate_mols]

        candidate_vecs = self.jtmpn(candidates, tree_mess)
        candidate_vecs = self.G_mean(candidate_vecs)
        mol_vec = mol_vec.squeeze()
        scores = torch.mv(candidate_vecs, mol_vec) * 20

        if prob_decode:
            probs = nn.Softmax()(scores.view(1, -1)).squeeze() + 1e-5  # prevent prob = 0
            # quick fix
            if len(probs.shape) == 0:
                probs = torch.tensor([probs])
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        for idx in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_atom_map = candidate_atom_maps[cand_idx[idx].item()]
            new_global_atom_map = copy.deepcopy(global_atom_map)

            for neighbor_id, ctr_atom_idx, neighbor_atom_idx in pred_atom_map:
                if neighbor_id == parent_nid:
                    continue
                new_global_atom_map[neighbor_id][neighbor_atom_idx] = new_global_atom_map[cur_node.nid][ctr_atom_idx]

            cur_mol = attach_mols(cur_mol, children, [], new_global_atom_map)  # parent is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            result = True
            for child_node in children:
                if child_node.is_leaf: continue
                cur_mol = self.dfs_assemble(tree_mess, mol_vec, all_nodes, cur_mol, new_global_atom_map, pred_atom_map,
                                            child_node, cur_node, prob_decode)
                if cur_mol is None:
                    result = False
                    break
            if result:
                return cur_mol

        return None