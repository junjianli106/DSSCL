import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from random import sample

from models import HEencoder

SMALL_NUM = np.log(1e-45)


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, args, base_encoder, dim=256, r=16384, m=0.999, T=0.1, mlp=False):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.args = args

        self.r = r
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = HEencoder.StainS(args, base_encoder, num_classes=dim)
        self.encoder_k = HEencoder.StainS(args, base_encoder, num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_index", torch.arange(0, r))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, index=None):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        if index is not None:
            index = concat_all_gather(index)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        if index is not None:
            self.queue_index[ptr: ptr + batch_size] = index
        ptr = (ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x1, x2):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x1.shape[0]
        x1_gather = concat_all_gather(x1)
        x2_gather = concat_all_gather(x2)
        batch_size_all = x1_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x1_gather[idx_this], x2_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def sample_neg_instance(self, im2cluster, im2second_cluster, silhouette, centroids, density, index):
        """
        mining based on the clustering results
        p1_select and p2_select
        """
        neg_index = self.queue_index.clone().detach()
        neg_proto_id = im2cluster[neg_index]
        neg_second_proto_id = im2second_cluster[neg_index]
        label = im2cluster[index]

        proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids.permute(1, 0))
        density = density.clamp(min=1e-3)
        proto_logit /= density

        # exclude negative samples with the same category as the anchor
        mask = neg_proto_id.unsqueeze(-1) != torch.arange(centroids.size(0)).cuda()
        proto_logit = proto_logit * mask
        proto_logit[proto_logit == 0] = SMALL_NUM

        logit = proto_logit.clone().detach().softmax(-1)
        p_sample = logit[:, label].t()

        mask = label.unsqueeze(-1) == neg_second_proto_id

        mask = silhouette[neg_index].unsqueeze(0).repeat(mask.size(0), 1) * mask
        mask[mask == 0] = 1

        mask = torch.distributions.bernoulli.Bernoulli(mask.clamp(0.0, 1.0).float()).sample()

        neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
        selected_mask = neg_sampler.sample()

        selected_mask = selected_mask * mask

        return selected_mask

    @torch.no_grad()
    def sample_neg_instance(self, im2cluster, im2second_cluster, silhouette, centroids, density, index):
        """
        mining based on the clustering results
        p1_select and p2_select
        """
        neg_index = self.queue_index.clone().detach()
        neg_proto_id = im2cluster[neg_index]
        neg_second_proto_id = im2second_cluster[neg_index]

        proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids.permute(1, 0))
        density = density.clamp(min=1e-3)
        proto_logit /= density
        label = im2cluster[index]
        logit = proto_logit.clone().detach().softmax(-1)
        p_sample = logit[:, label].t()

        mask = label.unsqueeze(-1) != neg_proto_id  # exclude negative samples with the same category as the anchor

        p_sample = p_sample * mask

        mask = label.unsqueeze(-1) == neg_second_proto_id

        mask = silhouette[neg_index].unsqueeze(0).repeat(mask.size(0), 1) * mask
        mask[mask == 0] = 1

        mask = torch.distributions.bernoulli.Bernoulli(mask.clamp(0.0, 1.0).float()).sample()

        neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
        selected_mask = neg_sampler.sample()

        selected_mask = selected_mask * mask

        return selected_mask

    # @torch.no_grad()
    # def sample_neg_instance(self, im2cluster, im2second_cluster, silhouette, centroids, density, index):
    #     """
    #     mining based on the clustering results
    #     p2_select
    #     """
    #     neg_index = self.queue_index.clone().detach()
    #     neg_second_proto_id = im2second_cluster[neg_index]
    #
    #     proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids.permute(1, 0))
    #     density = density.clamp(min=1e-3)
    #     proto_logit /= density
    #     label = im2cluster[index]
    #     proto_logit[proto_logit == 0] = SMALL_NUM
    #     logit = proto_logit.clone().detach().softmax(-1)
    #     p_sample = logit[:, label].t()
    #
    #     mask = label.unsqueeze(-1) == neg_second_proto_id
    #
    #     mask = silhouette[neg_index].unsqueeze(0).repeat(self.args.batch_size, 1) * mask
    #     mask[mask == 0] = 1
    #
    #     mask = torch.distributions.bernoulli.Bernoulli(mask.clamp(0.0, 1.0).float()).sample()
    #
    #     neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
    #     selected_mask = neg_sampler.sample()
    #
    #     selected_mask = selected_mask * mask
    #
    #     return selected_mask

    # @torch.no_grad()
    # def sample_neg_instance(self, im2cluster, centroids, density, index):
    #     """
    #     mining based on the clustering results
    #     p1_select
    #     """
    #     neg_index = self.queue_index.clone().detach()
    #     neg_proto_id = im2cluster[neg_index]
    #
    #     proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids.permute(1, 0))
    #     density = density.clamp(min=1e-3)
    #     proto_logit /= density
    #     label = im2cluster[index]
    #     logit = proto_logit.clone().detach().softmax(-1)
    #     p_sample = logit[:, label].t()
    #
    #     mask = label.unsqueeze(-1) != neg_proto_id
    #
    #     p_sample = p_sample * mask
    #
    #     neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
    #     selected_mask = neg_sampler.sample()
    #
    #     return selected_mask

    def forward(self, im_q_h, im_q_e, im_k_h=None, im_k_e=None, is_eval=False, cluster_result=None, index=None):
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

        if is_eval:
            k = self.encoder_k(im_q_h, im_q_e)
            k = nn.functional.normalize(k, dim=1)
            return k

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k_h, im_k_e, idx_unshuffle = self._batch_shuffle_ddp(im_k_h, im_k_e)

            k = self.encoder_k(im_k_h, im_k_e)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute query features
        q = self.encoder_q(im_q_h, im_q_e)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        if cluster_result is not None:
            inst_labels = []
            inst_logits = []
            for n, (im2cluster, im2second_cluster, prototypes, density, silhouette) in enumerate(
                    zip(cluster_result['im2cluster'], cluster_result['im2second_cluster'], cluster_result['centroids'],
                        cluster_result['density'], cluster_result['silhouette'])):
                # get negative instance
                selected_mask = self.sample_neg_instance(im2cluster, im2second_cluster, silhouette, prototypes, density,
                                                         index)

                l_neg = l_neg * selected_mask.clone().float()
                l_neg[l_neg == 0] = SMALL_NUM

                # logits: Nx(1+r)
                logits = torch.cat([l_pos, l_neg], dim=1)

                # apply temperature
                # count_zeros = torch.sum(selected_mask.clone() == 0, dim=1).unsqueeze(-1)  # 统计每行中0的个数
                # ratios = count_zeros.float() / selected_mask.shape[1]  # 计算比例
                # self.T = self.T * torch.exp(- 1.0 * ratios)
                logits /= self.T

                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

                inst_labels.append(labels)
                inst_logits.append(logits)

            # dequeue and enqueue
            self._dequeue_and_enqueue(k, index)

            return inst_logits, inst_labels
        else:
            # logits: Nx(1+r)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k, index)

            return logits, labels

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

# import torch
# import torch.nn as nn
# import numpy as np
# from random import sample
#
# from models import HEencoder
# from timm.models.layers import trunc_normal_
#
# SMALL_NUM = np.log(1e-45)
#
#
# class MoCo(nn.Module):
#     """
#     Build a MoCo model with: a query encoder, a key encoder, and a queue
#     https://arxiv.org/abs/1911.05722
#     """
#
#     def __init__(self, args, base_encoder, dim=256, r=16384, m=0.999, T=0.1, mlp=False):
#         """
#         dim: feature dimension (default: 128)
#         r: queue size; number of negative samples/prototypes (default: 16384)
#         m: momentum for updating key encoder (default: 0.999)
#         T: softmax temperature
#         mlp: whether to use mlp projection
#         """
#         super(MoCo, self).__init__()
#
#         self.args = args
#
#         self.r = r
#         self.m = m
#         self.T = T
#
#         # create the encoders
#         # num_classes is the output fc dimension
#         self.encoder_q = HEencoder.StainS(args, base_encoder, num_classes=dim)
#         self.encoder_k = HEencoder.StainS(args, base_encoder, num_classes=dim)
#
#         hidden_dim = self.encoder_q.fc.weight.shape[1]
#
#         self.instance_projector = nn.Sequential(
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, dim),
#         )
#         self.cluster_projector = nn.Sequential(
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 10),
#         )
#         trunc_normal_(self.cluster_projector[2].weight, std=0.02)
#         trunc_normal_(self.cluster_projector[5].weight, std=0.02)
#
#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data.copy_(param_q.data)  # initialize
#             param_k.requires_grad = False  # not update by gradient
#
#         # create the queue
#         self.register_buffer("queue", torch.randn(dim, r))
#         self.queue = nn.functional.normalize(self.queue, dim=0)
#
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
#         self.register_buffer("queue_index", torch.arange(0, r))
#
#     @torch.no_grad()
#     def _momentum_update_key_encoder(self):
#         """
#         Momentum update of the key encoder
#         """
#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
#
#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, keys, index=None):
#         # gather keys before updating queue
#         keys = concat_all_gather(keys)
#         if index is not None:
#             index = concat_all_gather(index)
#         batch_size = keys.shape[0]
#
#         ptr = int(self.queue_ptr)
#         assert self.r % batch_size == 0  # for simplicity
#
#         # replace the keys at ptr (dequeue and enqueue)
#         self.queue[:, ptr:ptr + batch_size] = keys.T
#         if index is not None:
#             self.queue_index[ptr: ptr + batch_size] = index
#         ptr = (ptr + batch_size) % self.r  # move pointer
#
#         self.queue_ptr[0] = ptr
#
#     @torch.no_grad()
#     def _batch_shuffle_ddp(self, x1, x2):
#         """
#         Batch shuffle, for making use of BatchNorm.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x1.shape[0]
#         x1_gather = concat_all_gather(x1)
#         x2_gather = concat_all_gather(x2)
#         batch_size_all = x1_gather.shape[0]
#
#         num_gpus = batch_size_all // batch_size_this
#
#         # random shuffle index
#         idx_shuffle = torch.randperm(batch_size_all).cuda()
#
#         # broadcast to all gpus
#         torch.distributed.broadcast(idx_shuffle, src=0)
#
#         # index for restoring
#         idx_unshuffle = torch.argsort(idx_shuffle)
#
#         # shuffled index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
#
#         return x1_gather[idx_this], x2_gather[idx_this], idx_unshuffle
#
#     @torch.no_grad()
#     def _batch_unshuffle_ddp(self, x, idx_unshuffle):
#         """
#         Undo batch shuffle.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x.shape[0]
#         x_gather = concat_all_gather(x)
#         batch_size_all = x_gather.shape[0]
#
#         num_gpus = batch_size_all // batch_size_this
#
#         # restored index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
#
#         return x_gather[idx_this]
#
#     @torch.no_grad()
#     def sample_neg_instance(self, im2cluster, im2second_cluster, silhouette, centroids, density, index):
#         """
#         mining based on the clustering results
#         p1_select and p2_select
#         """
#         neg_index = self.queue_index.clone().detach()
#         neg_proto_id = im2cluster[neg_index]
#         neg_second_proto_id = im2second_cluster[neg_index]
#
#         proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids.permute(1, 0))
#         density = density.clamp(min=1e-3)
#         proto_logit /= density
#         label = im2cluster[index]
#         proto_logit[proto_logit == 0] = SMALL_NUM
#         logit = proto_logit.clone().detach().softmax(-1)
#         p_sample = logit[:, label].t()
#
#         mask = label.unsqueeze(-1) != neg_proto_id  # exclude negative samples with the same category as the anchor
#
#         p_sample = p_sample * mask
#
#         mask = label.unsqueeze(-1) == neg_second_proto_id
#
#         mask = silhouette[neg_index].unsqueeze(0).repeat(self.args.batch_size, 1) * mask
#         mask[mask == 0] = 1
#
#         mask = torch.distributions.bernoulli.Bernoulli(mask.clamp(0.0, 1.0).float()).sample()
#
#         neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
#         selected_mask = neg_sampler.sample()
#
#         selected_mask = selected_mask * mask
#
#         return selected_mask
#
#     # @torch.no_grad()
#     # def sample_neg_instance(self, im2cluster, im2second_cluster, silhouette, centroids, density, index):
#     #     """
#     #     mining based on the clustering results
#     #     p2_select
#     #     """
#     #     neg_index = self.queue_index.clone().detach()
#     #     neg_second_proto_id = im2second_cluster[neg_index]
#     #
#     #     proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids.permute(1, 0))
#     #     density = density.clamp(min=1e-3)
#     #     proto_logit /= density
#     #     label = im2cluster[index]
#     #     proto_logit[proto_logit == 0] = SMALL_NUM
#     #     logit = proto_logit.clone().detach().softmax(-1)
#     #     p_sample = logit[:, label].t()
#     #
#     #     mask = label.unsqueeze(-1) == neg_second_proto_id
#     #
#     #     mask = silhouette[neg_index].unsqueeze(0).repeat(self.args.batch_size, 1) * mask
#     #     mask[mask == 0] = 1
#     #
#     #     mask = torch.distributions.bernoulli.Bernoulli(mask.clamp(0.0, 1.0).float()).sample()
#     #
#     #     neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
#     #     selected_mask = neg_sampler.sample()
#     #
#     #     selected_mask = selected_mask * mask
#     #
#     #     return selected_mask
#
#     # @torch.no_grad()
#     # def sample_neg_instance(self, im2cluster, centroids, density, index):
#     #     """
#     #     mining based on the clustering results
#     #     p1_select
#     #     """
#     #     neg_index = self.queue_index.clone().detach()
#     #     neg_proto_id = im2cluster[neg_index]
#     #
#     #     proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids.permute(1, 0))
#     #     density = density.clamp(min=1e-3)
#     #     proto_logit /= density
#     #     label = im2cluster[index]
#     #     logit = proto_logit.clone().detach().softmax(-1)
#     #     p_sample = logit[:, label].t()
#     #
#     #     mask = label.unsqueeze(-1) != neg_proto_id
#     #
#     #     p_sample = p_sample * mask
#     #
#     #     neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
#     #     selected_mask = neg_sampler.sample()
#     #
#     #     return selected_mask
#
#     def forward(self, im_q_h, im_q_e, im_k_h=None, im_k_e=None, is_eval=False, cluster_result=None, index=None):
#         """
#         Input:
#             im_q: a batch of query images
#             im_k: a batch of key images
#             is_eval: return momentum embeddings (used for clustering)
#             cluster_result: cluster assignments, centroids, and density
#             index: indices for training samples
#         Output:
#             logits, targets, proto_logits, proto_targets
#         """
#
#         if is_eval:
#             k = self.encoder_k(im_q_h, im_q_e)
#             k = nn.functional.normalize(k, dim=1)
#             return k
#
#         # compute key features
#         with torch.no_grad():  # no gradient to keys
#             self._momentum_update_key_encoder()  # update the key encoder
#
#             # shuffle for making use of BN
#             im_k_h, im_k_e, idx_unshuffle = self._batch_shuffle_ddp(im_k_h, im_k_e)
#
#             k = self.encoder_k(im_k_h, im_k_e)  # keys: NxC
#             k = nn.functional.normalize(k, dim=1)
#
#             # undo shuffle
#             k = self._batch_unshuffle_ddp(k, idx_unshuffle)
#
#         # compute query features
#         q = self.encoder_q(im_q_h, im_q_e)  # queries: NxC
#         q = nn.functional.normalize(q, dim=1)
#
#         # compute logits
#         # Einstein sum is more intuitive
#         # positive logits: Nx1
#         l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
#         # negative logits: Nxr
#         l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
#
#         if cluster_result is not None:
#             inst_labels = []
#             inst_logits = []
#             for n, (im2cluster, im2second_cluster, prototypes, density, silhouette) in enumerate(
#                     zip(cluster_result['im2cluster'], cluster_result['im2second_cluster'], cluster_result['centroids'],
#                         cluster_result['density'], cluster_result['silhouette'])):
#                 # get negative instance
#                 selected_mask = self.sample_neg_instance(im2cluster, im2second_cluster, silhouette, prototypes, density,
#                                                          index)
#
#                 l_neg = l_neg * selected_mask.clone().float()
#                 l_neg[l_neg == 0] = SMALL_NUM
#
#                 # logits: Nx(1+r)
#                 logits = torch.cat([l_pos, l_neg], dim=1)
#
#                 # apply temperature
#                 # count_zeros = torch.sum(selected_mask.clone() == 0, dim=1).unsqueeze(-1)  # 统计每行中0的个数
#                 # ratios = count_zeros.float() / selected_mask.shape[1]  # 计算比例
#                 # self.T = self.T * torch.exp(- 1.0 * ratios)
#                 logits /= self.T
#
#                 # labels: positive key indicators
#                 labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
#
#                 inst_labels.append(labels)
#                 inst_logits.append(logits)
#
#             # dequeue and enqueue
#             self._dequeue_and_enqueue(k, index)
#
#             return inst_logits, inst_labels
#         else:
#             # logits: Nx(1+r)
#             logits = torch.cat([l_pos, l_neg], dim=1)
#
#             # apply temperature
#             logits /= self.T
#
#             # labels: positive key indicators
#             labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
#
#             # dequeue and enqueue
#             self._dequeue_and_enqueue(k, index)
#
#             return logits, labels
#
# # utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#                       for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#
#     output = torch.cat(tensors_gather, dim=0)
#     return output
