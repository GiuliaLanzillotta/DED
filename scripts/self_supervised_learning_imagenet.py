""" 
Author: Giulia Lanzillotta
Date: Wed 1st of Nov

Self-supervised learning script: training a model on Cifar10 data using SimCLR.


python scripts/self_supervised_learning_imagenet.py --seed 11 --gpus_id 0 --batch_size 8  --checkpoints --notes SSL-imagenet-rn50 --wandb_project DataEfficientDistillation

Original paper: https://arxiv.org/pdf/2002.05709.pdf


"""

import importlib
import json
import math
import os
import socket
import sys
import time

internal_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(internal_path)
sys.path.append(internal_path + '/dataset_utils')
sys.path.append(internal_path + '/utils')


import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch
import numpy as np

from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageNet, ImageFolder, CIFAR10
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18, googlenet, efficientnet_b0, mobilenet_v3_large, resnext101_64x4d
import torchvision.transforms as transforms
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm

import shutil
from utils.args import add_management_args, add_rehearsal_args
from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar
from utils.stil_losses import *
from utils.nets import *

try:
    import wandb
except ImportError:
    wandb = None


def test(net, train_data_loader, test_data_loader, t):
    """ Following https://github.com/leftthomas/SimCLR/blob/master/main.py 
    
    test using weighted knn to find the most similar images' label to assign the test image
    """
    k = 200 # knn parameter
    c = 10 # cifar10 classes
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(train_data_loader, desc='Feature extracting'):
            features, _ = net(data.cuda(non_blocking=True))
            feature_bank.append(features)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(train_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            features, _ = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(features, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / t).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100



def setup_optimizerNscheduler(args, model, stud=False):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.optim_wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                            T_max=args.n_epochs, eta_min=args.lr) # so no reduction applied
        return optimizer, scheduler

        

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    path = base_path() + "/chkpts" + "/" + "imagenet" + "/" + "rn50-ssl/"
    if not os.path.exists(path): os.makedirs(path)
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(path+filename, path+'model_best.ckpt')

def load_checkpoint(best=False, filename='checkpoint.pth.tar', distributed=False):
    path = base_path() + "chkpts" + "/" + "imagenet" + "/" + "rn50-ssl/"
    if best: filepath = path + 'model_best.ckpt'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath)
          return checkpoint
    return None 

def parse_args(buffer=False):
    torch.set_num_threads(4)
    parser = ArgumentParser(description='script-experiment', allow_abbrev=False)
    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--checkpoints', action='store_true', help='Storing a checkpoint at every epoch. Loads a checkpoint if present.')
    parser.add_argument('--pretrained', action='store_true', help='Using a pre-trained network instead of training one.')
    parser.add_argument('--optim_wd', type=float, default=1e-6, help='optimizer weight decay.')
    parser.add_argument('--optim_adam', default=False, action='store_true', help='Using the Adam optimizer instead of SGD.')
    parser.add_argument('--optim_mom', type=float, default=0, help='optimizer momentum.')
    parser.add_argument('--optim_warmup', type=int, default=5, help='Number of warmup epochs.')
    parser.add_argument('--optim_nesterov', default=False, action='store_true', help='optimizer nesterov momentum.')
    parser.add_argument('--optim_cosineanneal', default=True, action='store_true', help='Enabling cosine annealing of learning rate..')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default = 512, help='Batch size.')
    parser.add_argument('--validate_subset', type=int, default=-1, 
                        help='If positive, allows validating on random subsets of the validation dataset during training.')
    parser.add_argument('--temperature', type=float, default=0.5, help='The temperature parameter for nll')

    add_management_args(parser)

    args = parser.parse_args()

    return args

class Model(nn.Module):
    # https://github.com/leftthomas/SimCLR/blob/master/model.py
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class ContrastiveTransformations(object):
    """ helper class to prepare the data loading such that we sample 
    two different, random augmentations for each image in the batch"""
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

def info_nce_loss(outputs, t):
        # [2*B, 2*B]
        batch_size = outputs.shape[0]//2
        sim_matrix = torch.exp(torch.mm(outputs, outputs.t().contiguous()) / t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(batch_size*2, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size*2, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(outputs[:batch_size] * outputs[batch_size:], dim=-1) / t)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        return loss
  

args = parse_args()
# Add uuid, timestamp and hostname for logging
args.conf_jobnum = str(uuid.uuid4())
args.conf_timestamp = str(datetime.datetime.now())
args.conf_host = socket.gethostname()

if args.seed is not None:
        set_random_seed(args.seed)


# dataset -> imagenet
imagenet_root = "/local/home/stuff/imagenet/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# transformations: Overall, for our experiments, we apply a set of 5 transformations following 
# the original SimCLR setup: random horizontal flip, crop-and-resize, color distortion, 
# random grayscale, and gaussian blur. 
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


train_dataset = ImageFolder(imagenet_root+'train', ContrastiveTransformations(train_transform, n_views=2))
another_train_dataset = ImageFolder(imagenet_root+'train', test_transform)
val_dataset = ImageFolder(imagenet_root+'val', test_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                          shuffle=True, num_workers=4, pin_memory=True)
another_train_loader = DataLoader(another_train_dataset, batch_size=args.batch_size, 
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                        shuffle=False, num_workers=4, pin_memory=False)


print("Dataset ready.")

# initialising the model
# model = resnet18(weights=None, num_classes=512*4)
# # following https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
# # we add an extra linear layer on top
# model.fc = nn.Sequential(
#             model.fc,  
#             nn.ReLU(inplace=True),
#             nn.Linear(4*512, 512)
#         )
model = Model(feature_dim=128)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"Network instantiated with {params} parameters")

setproctitle.setproctitle('{}_{}_{}'.format("resnet50", "ssl", "imagenet"))

print(args)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(d) for d in args.gpus_id])
if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if args.wandb_name is None: 
                name = str.join("-",["ssl", "imagenet", "rn18", args.conf_timestamp])
        else: name = args.wandb_name
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
                        name=name, notes=args.notes, config=vars(args)) 
        args.wandb_url = wandb.run.get_url()
device = get_device([0]) # returns the first device in the list
model.to(device)
progress_bar = ProgressBar(verbose=not args.non_verbose)

print(file=sys.stderr)

CHKPT_NAME = f'rn50-ssl-imagenet.ckpt' # obtaineed with seed = 11

model.train()
optimizer, scheduler = setup_optimizerNscheduler(args, model)
best_acc = 0.
start_epoch = 0
epoch=0
is_best = True


if args.checkpoints: # resuming training from the last point
        checkpoint = load_checkpoint(best=False, filename=CHKPT_NAME, distributed=False) 
        if checkpoint: 
                model.load_state_dict(checkpoint['state_dict'])
                model.to(device)
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                val_acc = checkpoint['best_acc']
                best_acc = val_acc

print("Starting training using SimCLR objective...")

for epoch in range(start_epoch, args.n_epochs):
        avg_loss = 0.0
        total = 0.0
        if args.debug_mode and epoch > 3:
                break
        for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                if args.debug_mode and i > 3: # only 3 batches in debug mode
                        break
                inputs, _ = data
                inputs = torch.cat(inputs, dim=0)
                inputs = inputs.to(device)
                features, outputs = model(inputs)
                loss = info_nce_loss(outputs, args.temperature)
                loss.backward()
                optimizer.step()

                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, 'SSL', loss.item())
                avg_loss += loss*(inputs.shape[0]//2)
                total += (inputs.shape[0]//2)

        test_acc_1, test_acc_5 = test(model, another_train_loader, val_loader, t=args.temperature)
        val_acc = test_acc_1
        avg_loss = avg_loss/total

        if scheduler is not None:
                scheduler.step()
        
        

        # best val accuracy -> selection bias on the validation set
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('\Train loss : {}'.format(round(avg_loss.item(), 2)), file=sys.stderr)
        print('\Val accuracy top-1: {} %'.format(round(test_acc_1, 2)), file=sys.stderr)
        print('\Val accuracy top-5 : {} %'.format(round(test_acc_5, 2)), file=sys.stderr)
        
        df = {'epoch_loss':avg_loss,
        'epoch_val_acctop1':test_acc_1,
        'epoch_val_acctop5':test_acc_5}
        wandb.log(df)

        if args.checkpoints and (is_best or epoch==args.n_epochs-1):
                save_checkpoint({
                'epoch': args.n_epochs + 1,
                'state_dict': model.state_dict(),
                'best_acc': val_acc,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict() if scheduler else None
                }, is_best, filename=CHKPT_NAME)
        
final_val_acc_D = val_acc

print("Training completed.")