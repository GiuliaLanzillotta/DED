
import os
from typing import Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torch.utils.data import Dataset
import torch

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path


class Cifar5MData: 

    def __init__(self, root: str):
        self.root = root
        self.loaded = False
    
    def load(self):
        '''
            Returns 5million synthetic samples.
            warning: returns as numpy array of unit8s, not torch tensors.
        '''

        nte = 10000 # num. of test samples to use (max 1e6)
        npart = 1000448
        X_tr = np.empty((5*npart, 32, 32, 3), dtype=np.uint8)
        Ys = []
        print('Loading CIFAR 5mil...')
        for i in range(5):
            z = np.load(os.path.join(self.root, f'cifar5m-part{i}.npz'))
            X_tr[i*npart: (i+1)*npart] = z['X']
            Ys.append(torch.tensor(z['Y']).long())
            print(f'Loaded part {i+1}/6')
        Y_tr = torch.cat(Ys)
        
        z = np.load(os.path.join(self.root, f'cifar5m-part5.npz')) # use the 6th million for test.
        print(f'Loaded part 6/6')
        
        X_te = z['X'][:nte]
        Y_te = torch.tensor(z['Y'][:nte]).long()
        
        self.loaded = True
        self.data_train = X_tr
        self.data_test = X_te
        self.targets_train = Y_tr
        self.targets_test = Y_te



class Cifar5M(Dataset):
    """
    Defines Cifar - 5 Million dataset. See https://github.com/preetum/cifar5m for reference. 
    """

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    def __init__(self, root: str, data:Cifar5MData,  train: bool = True, 
                 augmentations: Optional[nn.Module] = None) -> None:
        """Initialised a dataset from a data object (already loaded)"""
        self.root = root
        self.train = train

        assert data.loaded, "Load the data first"
        
        if augmentations is not None: 
            self.transform = torch.transforms.Compose([self.transform, augmentations])
        
        if train: 
            self.data = data.data_train
            self.targets = data.targets_train
        else: 
            self.data = data.data_test
            self.targets = data.targets_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = self.transform(img)


        return img, target