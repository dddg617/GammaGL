import argparse
from collections import defaultdict
import random
import numpy as np
import tensorlayerx as tlx
from gammagl.models.lightgcn import LightGCN
from gammagl.datasets.ml import MLDataset
from tensorlayerx.model import TrainOneStep, WithLoss
from sklearn.metrics import roc_auc_score, average_precision_score


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        return self._loss_fn(data)

class EdgeLoader:
    def __init__(self, edge_index, n_items, batch_size):
        self.edge_index = edge_index
        self.n_items = n_items
        self.batch_size = batch_size

    def __iter__(self):
        self.start = 0
        self.end = self.batch_size
        return self

    def __next__(self):
        if self.start > len(self.edge_index):
            raise StopIteration
        sub_edge_index = self.edge_index[self.start:self.end]
        self.start += self.batch_size
        self.end += self.batch_size

        map = defaultdict(set)
        l = list(zip(*sub_edge_index))

        for edge in sub_edge_index:
            map[edge[0].numpy()].add(edge[1].numpy())

        neg_items = []
        for edge in sub_edge_index:
            key = edge[0].numpy()
            i = 0
            # find 10 times, to avoid the situation that it can't find any neg_item
            while i < 10:
                i += 1
                if i == 10:
                    neg_items.append(0)
                else:
                    r = random.randint(0, self.n_items - 1)
                    if r not in map[key]:
                        neg_items.append(tlx.convert_to_tensor(r))
                        break
        return {'user_id': tlx.convert_to_tensor(l[0]), 'item_id': tlx.convert_to_tensor(l[1]),
                'neg_item_id': tlx.convert_to_tensor(neg_items, dtype=tlx.int64)}

def init_data(graph, n_items, batch_size):
    n_edges = len(graph.edge_index[0])
    train_pos = int(n_edges / 10 * 8)
    val_pos = int(n_edges / 10 + train_pos)

    train_data = list(zip(graph.edge_index[0][:train_pos], graph.edge_index[1][:train_pos]))
    val_data = list(zip(graph.edge_index[0][train_pos:val_pos], graph.edge_index[1][train_pos:val_pos]))
    test_data = list(zip(graph.edge_index[0][val_pos:], graph.edge_index[1][val_pos:]))

    train_batch = EdgeLoader(train_data, n_items, batch_size)
    val_batch = EdgeLoader(val_data, n_items, batch_size)
    test_batch = EdgeLoader(test_data, n_items, batch_size)

    return train_batch, val_batch, test_batch


def get_roc_score(u_emb, i_emb, edges):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.dot(u_emb, tlx.transpose(i_emb))

    users = edges['user_id']
    pos = edges['item_id']
    negs = edges['neg_item_id']

    preds = []
    preds_neg = []
    for i, _ in enumerate(users):
        preds.append(sigmoid(adj_rec[users[i], pos[i]]))
        preds_neg.append(sigmoid(adj_rec[users[i], negs[i]]))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def main(args):
    dataset = MLDataset(args.dataset_path,dataset_name=args.dataset)

    graph = dataset[0]

    train_data, val_data, test_data = init_data(graph, graph.n_items, args.batch_size)

    net = LightGCN(graph, args.hidden_dim)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)

    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, net.calculate_loss)

    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_ap_curr = 0
    for epoch in range(args.n_epoch):

        net.set_train()
        total_loss = 0
        for batch in train_data:
            train_loss = train_one_step(batch, None)
            total_loss += train_loss.item()

        net.set_eval()
        # link_prediction
        for batch in val_data:
            u_emb, i_emb = net(batch)

            roc_curr, ap_curr = get_roc_score(u_emb, i_emb, batch)

            print("Epoch [{:0>3d}] ".format(epoch + 1) \
                  + "  train loss: {:.4f}".format(total_loss) \
                  + "  val score: {:.4f}".format(ap_curr))

            if ap_curr > best_ap_curr:
                best_ap_curr = ap_curr
                net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    # if tlx.BACKEND == 'torch':
    #     net.to(batch['x'].device)
    net.set_eval()

    for batch in test_data:
        u_emb, i_emb = net(batch)

        roc_curr, ap_curr = get_roc_score(u_emb, i_emb, batch)
        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "   Test AUC score:  {:.4f}".format(roc_curr) \
              + "   Test AP score:  {:.4f}".format(ap_curr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=50, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimention of hidden layers1")
    parser.add_argument("--drop_rate", type=float, default=0., help="drop_rate")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--norm", type=str, default='none', help="how to apply the normalizer.")
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")  # 5e-4
    parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../../../data', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--batch_size", type=int, default=2048, help="size of a mini-batch")
    args = parser.parse_args()
    main(args)
