import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing

class BiGNNConv(MessagePassing):
    r"""Propagate a layer of Bi-interaction GNN
    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin1 = tlx.nn.Linear(in_channels=in_channels, out_channels=out_channels)
        self.lin2 = tlx.nn.Linear(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, edge_index, edge_weight):
        x_prop = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        x_trans = self.lin1(x_prop + x)
        x_inter = self.lin2(tlx.multiply(x_prop, x))
        return x_trans + x_inter

    def message(self, x_j, edge_weight):
        return tlx.reshape(edge_weight, (-1, 1)) * x_j