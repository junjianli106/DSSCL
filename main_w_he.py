import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from data_loader import BasicDataset
import torch.nn.functional as F
from models.HEresnet import resnet18

from tqdm import tqdm
import pcl.loader
import pcl.builder_he
import albumentations as A
from albumentations.pytorch import ToTensorV2


cuda_devices = '0,1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TORCH_DISTRIBUTED_DEBUG"] = 'INFO'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/home/xxxx/data/Kather_Multi_Class/NCT-CRC-HE-100K',
                    help='path to dataset')
parser.add_argument('--art_data', metavar='DIR', default='/home/xxxx/data/Kather_Multi_Class/Stain-Separation/train',
                    help='path to pos_dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2048, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--low-dim', default=256, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--pcl-r', default=65536, type=int,
                    help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')

parser.add_argument('--mlp', action='store_true', default=True,
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true', default=True,
                    help='use moco-v2/SimCLR data augmentation')
parser.add_argument('--cos', action='store_true', default=True,
                    help='use cosine lr schedule')

parser.add_argument('--num-cluster', default='20', type=str,
                    help='number of clusters')
parser.add_argument('--warmup-epoch', default=20, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--exp-dir', default='experiment_w_he', type=str,
                    help='experiment directory')
parser.add_argument('--stage', default='pre-train', type=str,
                    help='This code is the pre-training stage code')
parser.add_argument('--weight_orig', default=0.5, type=float,
                    help='The weight of the original image during image fusion')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.num_cluster = args.num_cluster.split(',')

    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = pcl.builder_he.MoCo(
        args,
        resnet18,
        args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.mlp)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    CE_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    CS_criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    criterion = {"CE_criterion": CE_criterion, "CS_criterion": CS_criterion}

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = args.data  # os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    eval_augmentation = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()])

    strong_augmentation = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.RandomBrightness(limit=0.2, p=0.75),  # 随机的亮度变化
            A.RandomContrast(limit=0.2, p=0.75),  # 随机的对比度
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),  # 高斯模糊
                A.GaussNoise(var_limit=(5.0, 30.0)),  # 加高斯噪声
            ], p=0.7),

            A.OneOf([
                    A.OpticalDistortion(distort_limit=1.0),  # 桶形 / 枕形畸变
                    A.GridDistortion(num_steps=5, distort_limit=1.),  # 网格畸变
                    A.ElasticTransform(alpha=3),  # 弹性变换
                ], p=0.4),

            A.CLAHE(clip_limit=4.0, p=0.7),  # 对输入图像应用限制对比度自适应直方图均衡化
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            # 随机改变图像的色调，饱和度，亮度。
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            # 应用仿射变换：平移、缩放、旋转
            A.Resize(224, 224),
            A.Cutout(max_h_size=int(224 * 0.375), max_w_size=int(224 * 0.375), num_holes=1, p=0.7),

            A.Normalize(),
            ToTensorV2()])

    weak_augmentation = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.RandomBrightness(limit=0.2, p=0.75),  # 随机的亮度变化
            A.RandomContrast(limit=0.2, p=0.75),  # 随机的对比度

            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            # 随机改变图像的色调，饱和度，亮度。
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
            # 应用仿射变换：平移、缩放、旋转
            A.Resize(224, 224),

            A.Normalize(),
            ToTensorV2()])

    train_dataset = BasicDataset(
        args,
        traindir,
        args.stage,
        pcl.loader.TwoCropsTransform([strong_augmentation, weak_augmentation]),
        args.art_data
    )

    eval_dataset = BasicDataset(
        args,
        traindir,
        args.stage,
        pcl.loader.TwoCropsTransform(eval_augmentation, is_eval=True),
        args.art_data
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size * 4, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):

        cluster_result = None
        if epoch >= args.warmup_epoch:
            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args)

            # placeholder for clustering result
            cluster_result = {'im2cluster':[], 'centroids':[], 'density':[], 'im2second_cluster':[], 'silhouette':[]}
            for num_cluster in args.num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(len(eval_dataset), dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster), args.low_dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
                cluster_result['im2second_cluster'].append(torch.zeros(len(eval_dataset), dtype=torch.long).cuda())
                cluster_result['silhouette'].append(torch.zeros(len(eval_dataset), dtype=torch.float).cuda())

            if args.gpu == 0:
                features[torch.norm(features, dim=1) > 1.5] /= 2 #account for the few samples that are computed twice
                features = features.numpy()
                cluster_result = run_kmeans(features, args)  #run kmeans clustering on master node
                # save the clustering result
                # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))

            dist.barrier()
            # broadcast clustering result
            for k, data_list in cluster_result.items():
                for data_tensor in data_list:
                    dist.broadcast(data_tensor, 0, async_op=False)

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, cluster_result)

        if (epoch + 1) % 20 == 0 and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                                               and args.rank % ngpus_per_node == 0)):
            filename = '{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir, epoch + 1)
            if (epoch + 1) == args.epochs:
                filename = '{}/checkpoint_{:04d}_{}_{}.pth.tar'.format(args.exp_dir, epoch + 1, args.weight_orig, args.num_cluster)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=filename)


def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    print('Total number of iterations: ', len(train_loader))
    for i, (images, index) in tqdm(enumerate(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)


        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images[2] = images[2].cuda(args.gpu, non_blocking=True)
            images[3] = images[3].cuda(args.gpu, non_blocking=True)

        output, target = model(im_q_h=images[0], im_q_e=images[1], im_k_h=images[2], im_k_e=images[3], cluster_result=cluster_result, index=index)

        if cluster_result is not None:
            loss_inst = 0

            for inst_out, inst_target in zip(output, target):
                loss_inst += criterion["CE_criterion"](inst_out, inst_target)
                acci = accuracy(inst_out, inst_target)[0]
                acc_inst.update(acci[0], images[0].size(0))

            # average loss across all sets of instances
            loss_inst /= len(args.num_cluster)
            loss = loss_inst

        else:
            # InfoNCE loss
            loss_inst = criterion["CE_criterion"](output, target)
            loss = loss_inst
            acc = accuracy(output, target)[0]
            acc_inst.update(acc[0], images[0].size(0))

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset), args.low_dim).cuda()
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            # images = images.cuda(non_blocking=True)
            images[0] = images[0].cuda(non_blocking=True)
            images[1] = images[1].cuda(non_blocking=True)
            feat = model(images[0], images[1], is_eval=True)
            features[index] = feat
    dist.barrier()
    dist.all_reduce(features, op=dist.ReduceOp.SUM)
    return features.cpu()


def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': [], 'im2second_cluster': [], 'silhouette': []}

    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]  # dim
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20  # clustering iterations
        clus.nredo = 5  # run the clustering this number of times, and keep the best centroids (selected according to clustering objective)
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        res.setTempMemory(0)
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 2)  # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]  # the cluster each image belongs to
        im2second_cluster = [int(n[1]) for n in I]

        silhouette = (D[:, 1] - D[:, 0]) / D[:, 1]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)  # Representation of each cluster

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

        # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        density = args.temperature * density / density.mean()  # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()
        im2second_cluster = torch.LongTensor(im2second_cluster).cuda()
        density = torch.Tensor(density).cuda()
        silhouette = torch.Tensor(silhouette).cuda()

        # silhouette = np.zeros((x.shape[0], k))
        #
        # im_rep2cluster = np.concatenate((x, I), axis=1)
        #
        # mask = im2cluster.unsqueeze(-1).repeat(1, k)
        # for i in range(mask.shape[1]):
        #     mask[:, i][mask[:, i] == i] = -1
        # mask[mask >= 0] = 1
        # mask[mask == -1] = 0
        #
        # for i in range(k):
        #     print(i)
        #     for j in range(i + 1, k):
        #         img_idx = np.where(np.logical_or(im_rep2cluster[:, -1] == i, im_rep2cluster[:, -1] == j))[0]
        #         images = np.concatenate((im_rep2cluster[im_rep2cluster[:, -1] == i, :], im_rep2cluster[im_rep2cluster[:, -1] == j, :]))
        #         labels = images[:, -1]
        #         silhouette[img_idx, i] = silhouette_samples(images[:, :-1], labels)[:, np.newaxis][:, 0]
        #         silhouette[img_idx, j] = silhouette_samples(images[:, :-1], labels)[:, np.newaxis][:, 0]
        #
        # silhouette = torch.from_numpy(silhouette).cuda() * mask

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)
        results['im2second_cluster'].append(im2second_cluster)
        results['silhouette'].append(silhouette)

    return results


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x


if __name__ == '__main__':
    # occumpy_mem(cuda_devices)
    main()
