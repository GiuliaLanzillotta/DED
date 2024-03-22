
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, OxfordIIITPet, Food101
from .cifar5m import Cifar5M, Cifar5MData
import numpy as np


def CIFAR100sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

def load_imagenet(augment=False):
    """Loads imagenet dataset"""

    augmentations = []
    if augment: augmentations = [transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip()]

    imagenet_root = "/local/home/stuff/imagenet/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


    train_transform = transforms.Compose(
                            augmentations + 
                            [
                                transforms.ToTensor(),
                                normalize,
                            ])
    inference_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])

    train_dataset = ImageFolder(imagenet_root+'train', train_transform if augment else inference_transform)
    val_dataset = ImageFolder(imagenet_root+'val', inference_transform)

    return train_dataset, val_dataset


def load_cifar10(augment=False):
    """Loads cifar10 dataset"""
    cifar10_root = '../continually/data/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    augmentations = []
    if augment: augmentations = [transforms.RandomCrop(32),
                                transforms.RandomHorizontalFlip()]

        
    train_dataset = CIFAR10(root=cifar10_root, train=True, 
                        transform=transforms.Compose(augmentations+[transforms.ToTensor(), normalize]))
    val_dataset = CIFAR10(root=cifar10_root, train=False, 
                        transform=transforms.Compose([transforms.ToTensor(), normalize]))

    return train_dataset, val_dataset


def load_cifar5m(augment=False):
    """Loading cifar5milion dataset"""

    cifar5m_root = "./data/CIFAR5M/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    augmentations = []
    if augment: augmentations = [transforms.RandomCrop(32),
                                transforms.RandomHorizontalFlip()]
        

    #cifar5m_root = "/local/home/stuff/cifar-5m/"
    data = Cifar5MData(root=cifar5m_root)
    data.load()
    train_dataset = Cifar5M(cifar5m_root, data, train=True, augmentations=transforms.Compose(augmentations))
    val_dataset = Cifar5M(cifar5m_root, data, train=False)

    return train_dataset, val_dataset

def load_food(augment=False):
    """Loads Food101 dataset"""

    # statistics source: https://github.com/hwchen2017/resnet_food101_cifar10_pytorch/blob/main/food101_resnet.py
    
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)
    food_root = '/local/home/stuff/food101/'
    normalize = transforms.Normalize(mean=_MEAN, std=_STD)

    standard_processing = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        normalize
    ]
    augmentations = []
    if augment: augmentations =  [
                    transforms.RandomResizedCrop(224),
	                transforms.RandomRotation(45),
                    transforms.RandomHorizontalFlip()
                ]
        
    train_dataset = Food101(root=food_root, download=True, 
                            transform=transforms.Compose(augmentations+standard_processing))
    val_dataset = Food101(root=food_root, split='test', download=True, 
                                transform=transforms.Compose(standard_processing))


    return train_dataset, val_dataset

def load_pet(augment=False):
    """Loads Oxford Pet III dataset"""
    # statistics source: https://github.com/Nahid01752/Oxford-IIIT-Pet_CNN/blob/main/Oxford-IIIT%20Pet_CNN.ipynb
    
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)
    pet_root = './data/oxford-pets/'
    normalize = transforms.Normalize(mean=_MEAN,
                                    std=_STD)

    standard_processing = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        normalize
    ]
    augmentations = []
    if augment: augmentations =  [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip()
                ]
        
    train_dataset = OxfordIIITPet(root=pet_root, download=True, 
                                  transform=transforms.Compose(augmentations+standard_processing))
    val_dataset = OxfordIIITPet(root=pet_root, split='test', download=True, 
                                transform=transforms.Compose(standard_processing))


    return train_dataset, val_dataset


def load_cifar100(augment=False):
    """Loads cifar100 dataset"""
    # statistics source: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py 
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    cifar100_root = '../continually/data/'
    normalize = transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,
                                    std=CIFAR100_TRAIN_STD)

    augmentations = []
    if augment: augmentations =  [
                    #transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15)
                ]
        
    train_dataset = CIFAR100(root=cifar100_root, train=True, 
                        transform=transforms.Compose(augmentations+[transforms.ToTensor(), normalize]))
    val_dataset = CIFAR100(root=cifar100_root, train=False, 
                        transform=transforms.Compose([transforms.ToTensor(), normalize]))

    return train_dataset, val_dataset

def load_dataset(name:str, augment=False):
    """Loading the dataset chosen. 
    Available options: imagenet, cifar5m."""

    if name=='imagenet': 
        return load_imagenet(augment)
    
    if name=='cifar10':
        return load_cifar10(augment)
    
    if name=='cifar100':
        return load_cifar100(augment)
    
    if name=='cifar5m':
        return load_cifar5m(augment)
    
    if name=="pet":
        return load_pet(augment)

    if name=="food":
        return load_food(augment)

    
    raise NotImplementedError