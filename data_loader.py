import random
import pcl.loader
import PIL
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2
import torchvision.transforms as transforms

# classes = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
classes = {'ADI': 0, 'DEB': 1, 'LYM': 2, 'MUC': 3, 'MUS': 4, 'NORM': 5, 'STR': 6, 'TUM': 7}

class BasicDataset(Dataset):
    def __init__(self, args, dataset_path, stage, transform, art_dataset_path=None):
        self.args = args
        self.dataset_path = dataset_path
        self.art_dataset_path = art_dataset_path
        self.images_list = []
        for file_name in os.listdir(self.dataset_path):
            if file_name != 'BACK':
                file_path = os.path.join(self.dataset_path, file_name)
                self.images_list.extend(os.listdir(file_path))

        self.transform = transform
        self.stage = stage

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        file_name = image_name.split('-')[0]
        image = cv2.imread(os.path.join(self.dataset_path, file_name, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.art_dataset_path is not None:
            image_H_name = image_name[:-4] + '_H_prime.png'
            image_H = cv2.imread(os.path.join(self.art_dataset_path + '/H_prime', image_H_name))
            image_H = cv2.cvtColor(image_H, cv2.COLOR_BGR2RGB)
            image_H = cv2.addWeighted(image, self.args.weight_orig, image_H, 1.0 - self.args.weight_orig, 0)

            image_E_name = image_name[:-4] + '_E_prime.png'
            image_E = cv2.imread(os.path.join(self.art_dataset_path + '/E_prime', image_E_name))
            image_E = cv2.cvtColor(image_E, cv2.COLOR_BGR2RGB)
            image_E = cv2.addWeighted(image, self.args.weight_orig, image_E, 1.0 - self.args.weight_orig, 0)

            if self.stage == 'pre-train':
                return self.transform(x_h=image_H, x_e=image_E), idx
            else:
                return self.transform(x_h=image_H, x_e=image_E), classes[file_name]
        else:
            image = PIL.Image.fromarray(image)

            if self.stage == 'pre-train':
                return self.transform(image), idx
            else:
                return self.transform(image), classes[file_name]


class PartialDataset(Dataset):
    def __init__(self, whole_dataset, indices):
        self.dataset = whole_dataset # all training data
        self.indices = indices # the index of the data selected for training

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def get_dataloader(args, art_dataset_path=None):
    # Data loading code
    traindir = os.path.join(args.data, 'NCT-CRC-HE-100K')
    valdir = os.path.join(args.data, 'NCT-CRC-HE-100K')
    testdir = os.path.join(args.data, 'CRC-VAL-HE-7K')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if art_dataset_path is not None:
        train_dataset = BasicDataset(
            args,
            traindir,
            args.stage,
            pcl.loader.TwoCropsTransform(augmentation, is_eval=True),
            os.path.join(art_dataset_path, 'train')
        )

        val_dataset = BasicDataset(
            args,
            valdir,
            args.stage,
            pcl.loader.TwoCropsTransform(augmentation, is_eval=True),
            os.path.join(art_dataset_path, 'train')
        )

        test_dataset = BasicDataset(
            args,
            testdir,
            args.stage,
            pcl.loader.TwoCropsTransform(augmentation, is_eval=True),
            os.path.join(art_dataset_path, 'test')
        )

    else:
        train_dataset = BasicDataset(
            args,
            traindir,
            args.stage,
            augmentation
        )

        val_dataset = BasicDataset(
            args,
            valdir,
            args.stage,
            augmentation
        )

        test_dataset = BasicDataset(
            args,
            testdir,
            args.stage,
            augmentation
        )

    n_train = len(train_dataset)
    indices = list(range(n_train))
    n_valid = int(args.val_split * n_train)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[n_valid:], indices[:n_valid]
    train_idx = np.random.choice(train_idx, int(args.labeled_train * len(train_idx))).tolist()

    train_dataset = PartialDataset(train_dataset, train_idx)
    val_dataset = PartialDataset(val_dataset, val_idx)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_sampler