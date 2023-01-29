# -*- coding: utf-8 -*-
r"""
NCL
################################################
Reference:
    Zihan Lin*, Changxin Tian*, Yupeng Hou*, Wayne Xin Zhao. "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning." in WWW 2022.
"""
import tensorlayerx as tlx
from gammagl.layers.conv import LightGCNConv
from gammagl.models.loss import BPRLoss, EmbLoss


class NCL(tlx.nn.Module):

    def __init__(self, config, dataset):
        super(NCL, self).__init__(config, dataset)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type: the embedding size of the base model
        self.n_layers = config['n_layers']          # int type: the layer num of the base model
        self.reg_weight = config['reg_weight']      # float32 type: the weight decay for l2 normalization

        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.hyper_layers = config['hyper_layers']

        self.alpha = config['alpha']

        self.proto_reg = config['proto_reg']
        self.k = config['num_clusters']

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
        # self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

    def e_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        import faiss
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        # centroids = torch.Tensor(cluster_cents).to(self.device)
        # centroids = F.normalize(centroids, p=2, dim=1)
        centroids = tlx.convert_to_tensor(cluster_cents)
        centroids = tlx.l2_normalize(centroids, axis = 1)
        
        node2cluster = tlx.squeeze(tlx.convert_to_tensor(I))

        # node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = tlx.concat([user_embeddings, item_embeddings], axis=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers * 2)):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = tlx.stack(embeddings_list[:self.n_layers + 1], axis=1)
        lightgcn_all_embeddings = tlx.reduce_mean(lightgcn_all_embeddings, axis=1)

        user_all_embeddings, item_all_embeddings = tlx.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def ProtoNCE_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = tlx.split(node_embedding, [self.n_users, self.n_items])

        user_embeddings = tlx.gather(user_embeddings_all, user)     # [B, e]
        norm_user_embeddings = tlx.l2_normalize(user_embeddings, axis=1)

        user2cluster = tlx.gather(self.user_2cluster, user)     # [B,]
        user2centroids = tlx.gather(self.user_centroids, user2cluster)   # [B, e]
        pos_score_user = tlx.reduce_sum(tlx.multiply(norm_user_embeddings, user2centroids), axis = 1)
        pos_score_user = tlx.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = tlx.matmul(norm_user_embeddings, tlx.transpose(self.user_centroids, [1, 0]))
        ttl_score_user = tlx.reduce_sum(tlx.exp(ttl_score_user / self.ssl_temp), axis=1)

        proto_nce_loss_user = -tlx.reduce_sum(tlx.log(pos_score_user / ttl_score_user))

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = tlx.l2_normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = tlx.reduce_sum(tlx.multiply(norm_item_embeddings, item2centroids), axis=1)
        pos_score_item = tlx.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = tlx.matmul(norm_item_embeddings, tlx.transpose(self.item_centroids, [1, 0]))
        ttl_score_item = tlx.reduce_sum(tlx.exp(ttl_score_item / self.ssl_temp), axis=1)
        proto_nce_loss_item = -tlx.reduce_sum(tlx.log(pos_score_item / ttl_score_item))

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = tlx.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = tlx.split(previous_embedding, [self.n_users, self.n_items])

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = tlx.l2_normalize(current_user_embeddings)
        norm_user_emb2 = tlx.l2_normalize(previous_user_embeddings)
        norm_all_user_emb = tlx.l2_normalize(previous_user_embeddings_all)
        pos_score_user = tlx.reduce_sum(tlx.multiply(norm_user_emb1, norm_user_emb2), axis=1)
        ttl_score_user = tlx.matmul(norm_user_emb1, tlx.transpose(norm_all_user_emb, [1, 0]))
        pos_score_user = tlx.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = tlx.reduce_sum(tlx.exp(ttl_score_user / self.ssl_temp), axis=1)

        ssl_loss_user = -tlx.reduce_sum(tlx.log(pos_score_user / ttl_score_user))

        current_item_embeddings = tlx.gather(current_item_embeddings, item)
        previous_item_embeddings = tlx.gather(previous_item_embeddings_all, item)
        norm_item_emb1 = tlx.l2_normalize(current_item_embeddings)
        norm_item_emb2 = tlx.l2_normalize(previous_item_embeddings)
        norm_all_item_emb = tlx.l2_normalize(previous_item_embeddings_all)
        pos_score_item = tlx.reduce_sum(tlx.multiply(norm_item_emb1, norm_item_emb2), axis=1)
        ttl_score_item = tlx.matmul(norm_item_emb1, tlx.transpose(norm_all_item_emb, [1, 0]))
        pos_score_item = tlx.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = tlx.reduce_sum(tlx.exp(ttl_score_item / self.ssl_temp), axis=1)

        ssl_loss_item = -tlx.reduce_sum(tlx.log(pos_score_item / ttl_score_item))

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        center_embedding = embeddings_list[0]
        context_embedding = embeddings_list[self.hyper_layers * 2]

        ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, user, pos_item)
        proto_loss = self.ProtoNCE_loss(center_embedding, user, pos_item)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = tlx.reduce_sum(tlx.mul(u_embeddings, pos_embeddings), axis=1)
        neg_scores = tlx.reduce_sum(tlx.mul(u_embeddings, neg_embeddings), axis=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss + self.reg_weight * reg_loss, ssl_loss, proto_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = tlx.reduce_sum(tlx.multiply(u_embeddings, i_embeddings), axis=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = tlx.matmul(u_embeddings, tlx.transpose(self.restore_item_e, [1, 0]))

        return tlx.reshape(scores, -1)