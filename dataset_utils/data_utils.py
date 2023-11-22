
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from .cifar5m import Cifar5M, Cifar5MData


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

    train_dataset = ImageFolder(imagenet_root+'train', train_transform)
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

def load_cifar100(augment=False):
    """Loads cifar10 dataset"""
    cifar100_root = '../continually/data/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    augmentations = []
    if augment: augmentations = [transforms.RandomCrop(32),
                                transforms.RandomHorizontalFlip()]

        
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

    
    raise NotImplementedError