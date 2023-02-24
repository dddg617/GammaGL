import tensorlayerx as tlx


class BPRLoss(tlx.nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking
    Args:
        - gamma(float): Small value to avoid division by zero
    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.
    Examples::
        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = tlx.reduce_mean(-tlx.log(self.gamma + tlx.sigmoid(pos_score - neg_score)))
        return loss

class EmbLoss(tlx.nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        # device = tlx.get_device(embeddings[-1])
        if require_pow:
            emb_loss = tlx.zeros(1)
            for embedding in embeddings:
                emb_loss += tlx.pow(
                    self.normalize(embedding, p=self.norm), self.norm
                )
            emb_loss /= tlx.get_tensor_shape(embeddings[-1])[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = tlx.zeros(1)
            for embedding in embeddings:
                emb_loss += self.normalize(embedding, p=self.norm)
            emb_loss /= tlx.get_tensor_shape(embeddings[-1])[0]
            return emb_loss
    def normalize(self, input, p):
        return tlx.reduce_sum(input ** p) ** (1/p)
