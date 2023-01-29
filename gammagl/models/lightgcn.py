import numpy as np
import tensorlayerx as tlx

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

    def __init__(self, embedding_size, n_layers, reg_weight, require_pow, user_id, item_id, neg_item_id, n_users, n_items):
        super(LightGCN, self).__init__()

        # load parameters info
        self.latent_dim = embedding_size
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.require_pow = require_pow
        self.n_users = n_users
        self.n_items = n_items
        self.USER_ID = user_id
        self.ITEM_ID = item_id
        self.NEG_ITEM_ID = neg_item_id

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
        self.apply(tlx.nn.xavier_uniform)
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

    def forward(self, edge_index, edge_weight = None):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
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

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

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

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = tlx.reduce_sum(tlx.multiply(u_embeddings, i_embeddings), axis=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = tlx.matmul(u_embeddings, tlx.transpose(self.restore_item_e.transpose, [1, 0]))

        return tlx.reshape(scores, -1)