import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'torch'
import argparse
from gammagl.datasets import Planetoid
from gammagl.layers.conv import GATConv
from gammagl.utils import add_self_loops, mask_to_index
import torch
import torch.nn.functional as F
import time
import tensorlayerx as tlx
import torch._dynamo
torch._dynamo.reset()

class GATModel(tlx.nn.Module):

    def __init__(self,feature_dim,
                 hidden_dim,
                 num_class,
                 heads,
                 drop_rate,
                 name=None,
                 ):
        super().__init__(name=name)
        
        self.conv1 = GATConv(
            in_channels=feature_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout_rate=drop_rate,
            concat=True
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=num_class,
            heads=heads,
            dropout_rate=drop_rate,
            concat=False
        )

        self.elu = tlx.layers.ELU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index, num_nodes):
        x = self.dropout(x)
        x = self.conv1(x, edge_index, num_nodes)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, num_nodes)
        x = self.elu(x)
        return x


def parameters(model, recurse = True):
    for _, param in model.named_parameters(recurse = recurse):
        yield param

def main(args):
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)

    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    model = GATModel(feature_dim=dataset.num_node_features,
                   hidden_dim=args.hidden_dim,
                   num_class=dataset.num_classes,
                   heads = args.heads,
                   drop_rate=args.drop_rate,
                   name="GAT")
    model = torch.compile(model, mode="reduce-overhead", dynamic=False)
    optimizer = torch.optim.Adam(params=parameters(model), lr=args.lr, weight_decay=args.l2_coef)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        # "edge_weight": edge_weight,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    start = time.time()
    for epoch in range(args.n_epoch):
        model.set_train()
        logits = model(data['x'], data['edge_index'], data['num_nodes'])
        train_logits = logits[data['train_idx']]
        train_y = data['y'][data['train_idx']]
        loss = F.cross_entropy(train_logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.set_eval()
        logits = model(data['x'], data['edge_index'], data['num_nodes'])
        val_logits = logits[data['val_idx']]
        val_y = data['y'][data['val_idx']]
        pred = torch.argmax(val_logits, dim=1)
        val_acc = (float((val_y==pred).sum()) / float(len(data['val_idx'])))

        # print("Epoch [{:0>3d}] ".format(epoch+1)\
        #       + "  train loss: {:.4f}".format(loss.item())\
        #       + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path+model.name+".npz", format='npz_dict')
    end = time.time()
    
    print("Total Time: ", end-start)
    model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')

    model.set_eval()
    logits = model(data['x'], data['edge_index'], data['num_nodes'])
    test_logits = logits[data['test_idx']]
    test_y = data['y'][data['test_idx']]
    pred = torch.argmax(test_logits, dim=1)
    test_acc = (float((test_y==pred).sum()) / float(len(data['test_idx'])))
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=1000, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=256, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.4, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument("--heads", type=int, default=32, help="number of heads for stablization")
    parser.add_argument('--dataset', type=str, default='pubmed', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--gpu", type=int, default=1)
    
    args = parser.parse_args()
    if args.gpu >= 0:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(args.gpu)

    main(args)