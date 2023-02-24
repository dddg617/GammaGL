import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing


class LightGCNConv(MessagePassing):
    def __init__(self, dim):
        super(LightGCNConv, self).__init__()
        self.dim = dim

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(x, edge_index, edge_weight=edge_weight)

    def message(self, x, edge_index, edge_weight):
        msg = tlx.gather(x, edge_index[0, :])
        return tlx.reshape(edge_weight, (-1, 1)) * msg

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)
