r"""
NGCF
################################################
Reference:
    Xiang Wang et al. "Neural Graph Collaborative Filtering." in SIGIR 2019.
Reference code:
    https://github.com/xiangwang1223/neural_graph_collaborative_filtering
"""
import tensorlayerx as tlx
from gammagl.layers.conv import BiGNNConv
from gammagl.models import BPRLoss, EmbLoss
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.utils import dropout_adj

# from recbole.model.init import xavier_normal_initialization
# from recbole.model.loss import BPRLoss, EmbLoss
# from recbole.utils import InputType

# from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
# from recbole_gnn.model.layers import BiGNNConv


class NGCF(tlx.nn.Module):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """

    def __init__(self, config, dataset):
        super(NGCF, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size_list = config['hidden_size_list']
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = config['node_dropout']
        self.message_dropout = config['message_dropout']
        self.reg_weight = config['reg_weight']

        # define layers and loss
        self.user_embedding = tlx.nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = tlx.nn.Embedding(self.n_items, self.embedding_size)
        self.GNNlayers = tlx.nn.ModuleList()
        for input_size, output_size in zip(self.hidden_size_list[:-1], self.hidden_size_list[1:]):
            self.GNNlayers.append(BiGNNConv(input_size, output_size))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        # self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = tlx.concat([user_embeddings, item_embeddings], axis=0)
        return ego_embeddings

    def forward(self):
        if self.node_dropout == 0:
            edge_index, edge_weight = self.edge_index, self.edge_weight
        else:
            edge_index, edge_weight = dropout_adj(edge_index=self.edge_index, edge_attr=self.edge_weight, p=self.node_dropout)

        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(all_embeddings, edge_index, edge_weight)
            all_embeddings = tlx.nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = tlx.nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = tlx.l2_normalize(all_embeddings, axis=1)
            embeddings_list += [all_embeddings]  # storage output embedding of each layer
        ngcf_all_embeddings = tlx.concat(embeddings_list, axis=1)

        user_all_embeddings, item_all_embeddings = tlx.split(ngcf_all_embeddings, [self.n_users, self.n_items])

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
        pos_embeddings = tlx.gather(item_all_embeddings, pos_item)
        neg_embeddings = tlx.gather(item_all_embeddings, neg_item)

        pos_scores = tlx.reduce_sum(tlx.multiply(u_embeddings, pos_embeddings), axis=1)
        neg_scores = tlx.reduce_sum(tlx.multiply(u_embeddings, neg_embeddings), axis=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)  # calculate BPR Loss

        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)  # L2 regularization of embeddings

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = tlx.gather(user_all_embeddings, user)
        i_embeddings = tlx.gather(item_all_embeddings, item)
        scores = tlx.reduce_sum(tlx.multiply(u_embeddings, i_embeddings), axis=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = tlx.matmul(u_embeddings, tlx.transpose(self.restore_item_e, [1, 0]))

        return tlx.reshape(scores, -1)