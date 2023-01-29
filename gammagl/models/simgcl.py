# -*- coding: utf-8 -*-
r"""
SimGCL
################################################
Reference:
    Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, Quoc Viet Hung Nguyen. "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation." in SIGIR 2022.
"""

import numpy as np
import tensorlayerx as tlx
from gammagl.models import LightGCN
# import torch
# import torch.nn.functional as F

# from recbole_gnn.model.general_recommender import LightGCN


class SimGCL(LightGCN):
    def __init__(self, config, dataset):
        super(SimGCL, self).__init__(config, dataset)

        self.cl_rate = config['lambda']
        self.eps = config['eps']
        self.temperature = config['temperature']

    def forward(self, perturbed=False):
        all_embs = self.get_ego_embeddings()
        embeddings_list = []

        for layer_idx in range(self.n_layers):
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)
            if perturbed:
                random_noise = tlx.convert_to_tensor(np.random.rand(tlx.get_tensor_shape(all_embs)[0], 
                                                                    tlx.get_tensor_shape(all_embs)[1]))
                # random_noise = torch.rand_like(all_embs, device=all_embs.device)
                all_embs = all_embs + tlx.sign(all_embs) * tlx.l2_normalize(random_noise, xis=-1) * self.eps
            embeddings_list.append(all_embs)
        lightgcn_all_embeddings = tlx.stack(embeddings_list, axis=1)
        lightgcn_all_embeddings = tlx.reduce_mean(lightgcn_all_embeddings, axis=1)

        user_all_embeddings, item_all_embeddings = tlx.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_cl_loss(self, x1, x2):
        x1, x2 = tlx.l2_normalize(x1, axis=-1), tlx.l2_normalize(x2, axis=-1)
        pos_score = tlx.reduce_sum((x1 * x2), axis=-1)
        pos_score = tlx.exp(pos_score / self.temperature)
        ttl_score = tlx.matmul(x1, tlx.transpose(x2, [1, 0]))
        ttl_score = tlx.reduce_sum(tlx.exp(ttl_score / self.temperature), axis=1)
        return -tlx.reduce_sum(tlx.log(pos_score / ttl_score))

    def calculate_loss(self, interaction):
        loss = super().calculate_loss(interaction)

        # [Bug] May get an error using np.unique
        user = tlx.convert_to_tensor(np.unique(interaction[self.USER_ID]))
        pos_item = tlx.convert_to_tensor(np.unique(interaction[self.ITEM_ID]))

        perturbed_user_embs_1, perturbed_item_embs_1 = self.forward(perturbed=True)
        perturbed_user_embs_2, perturbed_item_embs_2 = self.forward(perturbed=True)

        user_cl_loss = self.calculate_cl_loss(perturbed_user_embs_1[user], perturbed_user_embs_2[user])
        item_cl_loss = self.calculate_cl_loss(perturbed_item_embs_1[pos_item], perturbed_item_embs_2[pos_item])

        return loss + self.cl_rate * (user_cl_loss + item_cl_loss)