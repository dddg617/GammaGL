import numpy as np
import tensorlayerx as tlx
from gammagl.utils import degree

from gammagl.mpops import segment_sum, unsorted_segment_sum
from gammagl.layers.conv import LightGCNConv
from gammagl.models import BPRLoss, EmbLoss


class LightGCN(tlx.nn.Module):
    r"""LightGCN is a GCN-based recommender model, implemented via PyG.
    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly 
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.
    We implement the model following the original author with a pairwise training mode.
    """

    # def __init__(self, embedding_size, n_layers, reg_weight, require_pow, user_id, item_id, neg_item_id, n_users,
    #              n_items):
    def __init__(self, graph, embedding_size=128, n_layers=2, reg_weight=1e-05, require_pow=True):
        super(LightGCN, self).__init__()

        # load parameters info
        self.latent_dim = embedding_size
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.require_pow = require_pow
        self.n_users = graph.n_users
        self.n_items = graph.n_items

        def get_norm_edge_weight(edge_index):
            row = edge_index[0, :]
            col = edge_index[1, :]

            edge_index1 = tlx.stack([row, col])
            edge_index2 = tlx.stack([col, row])
            edge_index = tlx.concat([edge_index1, edge_index2], axis=1)

            deg = degree(edge_index[0, :], num_nodes=graph.n_users + graph.n_items)

            norm_deg = 1. / tlx.sqrt(tlx.where(deg == 0, tlx.ones([1]), deg))
            # edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

            edge_weight = tlx.gather(norm_deg, edge_index[0]) * tlx.gather(norm_deg, edge_index[1])

            return edge_index, edge_weight

        self.edge_index, self.edge_weight = get_norm_edge_weight(graph.edge_index)

        self.USER_ID = 'user_id'
        self.ITEM_ID = 'item_id'
        self.NEG_ITEM_ID = 'neg_item_id'

        # define layers and loss
        self.user_embedding = tlx.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = tlx.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        # self.apply(tlx.nn.xavier_uniform)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.embeddings
        item_embeddings = self.item_embedding.embeddings
        ego_embeddings = tlx.concat([user_embeddings, item_embeddings], axis=0)
        return ego_embeddings

    def get_all_embedding(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = tlx.stack(embeddings_list, axis=1)
        lightgcn_all_embeddings = tlx.reduce_mean(lightgcn_all_embeddings, axis=1)

        user_all_embeddings, item_all_embeddings = tlx.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.get_all_embedding()

        u_embeddings = tlx.gather(user_all_embeddings, user)
        pos_embeddings = tlx.gather(item_all_embeddings, pos_item)
        neg_embeddings = tlx.gather(item_all_embeddings, neg_item)

        # calculate BPR Loss
        pos_scores = tlx.reduce_sum(tlx.multiply(u_embeddings, pos_embeddings), axis=1)
        neg_scores = tlx.reduce_sum(tlx.multiply(u_embeddings, neg_embeddings), axis=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def forward(self, interaction):
        user_idx = interaction[self.USER_ID]
        item_idx = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.get_all_embedding()

        u_embeddings = tlx.gather(user_all_embeddings, user_idx)
        i_embeddings = tlx.gather(item_all_embeddings, item_idx)

        return u_embeddings, i_embeddings
