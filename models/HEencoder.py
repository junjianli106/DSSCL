import torch
import torch.nn as nn

class StainS(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, args, base_encoder, num_classes=256):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature
        mlp: whether to use mlp projection
        """
        super(StainS, self).__init__()

        self.args = args

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_h = base_encoder()
        self.encoder_e = base_encoder()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        dim_mlp = 1024 if self.args.arch[-2:] in ['18'] else 4096

        self.fc = nn.Linear(dim_mlp, num_classes)

    def forward(self, im_h, im_e):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """

        h = self.avgpool(self.encoder_h(im_h))
        e = self.avgpool(self.encoder_e(im_e))

        embedding = torch.cat([h, e], dim=1)

        embedding = torch.flatten(embedding, 1)

        embedding = self.fc(embedding)

        return embedding



