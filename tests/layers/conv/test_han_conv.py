import tensorlayerx as tlx

from gammagl.layers.conv import HANConv


def test_han_conv():
    x_dict = {
        'author': tlx.random_normal((6, 16), dtype = tlx.float32),
        'paper': tlx.random_normal((5, 12), dtype = tlx.float32)
    }
    edge1 = tlx.random_uniform((2, 7), 0, 6, dtype = tlx.int64)
    edge2 = tlx.random_uniform((2, 4), 0, 5, dtype = tlx.int64)
    edge3 = tlx.random_uniform((2, 5), 0, 4, dtype = tlx.int64)
    edge_index_dict = {
        ('author', 'metapath0', 'author'): edge1,
        ('paper', 'metapath1', 'paper'): edge2,
        ('paper', 'metapath2', 'paper'): edge3,
    }
    num_nodes_dict = {
        'author': 6,
        'paper': 5
    }
    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    in_channels = {'author': 16, 'paper': 12}

    conv = HANConv(in_channels, 16, metadata, heads=2)
    out_dict1 = conv(x_dict, edge_index_dict, num_nodes_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].shape == (6, 16)
    assert out_dict1['paper'].shape == (5, 16)

    # non zero dropout
    conv = HANConv(in_channels, 16, metadata, heads=2, dropout_rate=0.1)
    out_dict1 = conv(x_dict, edge_index_dict, num_nodes_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].shape == (6, 16)
    assert out_dict1['paper'].shape == (5, 16)

def test_han_conv_empty_tensor():
    x_dict = {
        'author': tlx.random_normal((6, 16), dtype = tlx.float32),
        'paper': tlx.random_uniform((0, 12), 0, 0, dtype = tlx.float32),
    }
    edge_index_dict = {
        ('paper', 'to', 'author'): tlx.random_uniform((2, 0), 0, 0, dtype = tlx.int64),
        ('author', 'to', 'paper'): tlx.random_uniform((2, 0), 0, 0, dtype = tlx.int64),
        ('paper', 'to', 'paper'): tlx.random_uniform((2, 0), 0, 0, dtype = tlx.int64),
    }
    num_nodes_dict = {
        'author': 6,
        'paper': 0
    }
    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    in_channels = {'author': 16, 'paper': 12}
    conv = HANConv(in_channels, 16, metadata, heads=2)

    out_dict = conv(x_dict, edge_index_dict, num_nodes_dict)
    assert len(out_dict) == 2
    assert out_dict['author'].shape == (6, 16)
    assert tlx.all(out_dict['author'] == 0)
    assert out_dict['paper'].shape == (0, 16)