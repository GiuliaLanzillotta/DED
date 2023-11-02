""" 
Author: Giulia Lanzillotta
Date: Wed 1st of Nov

Self-supervised learning script: training a model on some data using SSL. 


python scripts/self_supervised_learning.py --validate_subset 2000 --seed 11 --gpus_id 1 --batch_size 128  --checkpoints --notes SSL-imagenet-rn18 --wandb_project DataEfficientDistillation


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
sys.path.append(internal_path + '/datasets')
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


import shutil
from utils.args import add_management_args, add_rehearsal_args
from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar
from utils.stil_losses import *
from utils.nets import *
from datasets.data_utils import load_dataset

try:
    import wandb
except ImportError:
    wandb = None



def setup_optimizerNscheduler(args, model, stud=False):
        optimizer = torch.optim.AdamW(model.parameters(), 
                        lr=args.lr, 
                        weight_decay=args.optim_wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                            T_max=args.n_epochs, eta_min=args.lr/50)
        return optimizer, scheduler

        

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    path = base_path() + "/chkpts" + "/" + "imagenet" + "/" + "rn18-ssl/"
    if not os.path.exists(path): os.makedirs(path)
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(path+filename, path+'model_best.ckpt')

def load_checkpoint(best=False, filename='checkpoint.pth.tar', distributed=False):
    path = base_path() + "chkpts" + "/" + "imagenet" + "/" + "rn18-ssl/"
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
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--checkpoints', action='store_true', help='Storing a checkpoint at every epoch. Loads a checkpoint if present.')
    parser.add_argument('--pretrained', action='store_true', help='Using a pre-trained network instead of training one.')
    parser.add_argument('--optim_wd', type=float, default=1e-4, help='optimizer weight decay.')
    parser.add_argument('--optim_adam', default=False, action='store_true', help='Using the Adam optimizer instead of SGD.')
    parser.add_argument('--optim_mom', type=float, default=0, help='optimizer momentum.')
    parser.add_argument('--optim_warmup', type=int, default=5, help='Number of warmup epochs.')
    parser.add_argument('--optim_nesterov', default=False, action='store_true', help='optimizer nesterov momentum.')
    parser.add_argument('--optim_cosineanneal', default=True, action='store_true', help='Enabling cosine annealing of learning rate..')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--n_epochs_stud', type=int, default=30, help='Number of student epochs.')
    parser.add_argument('--batch_size', type=int, default = 256, help='Batch size.')
    parser.add_argument('--validate_subset', type=int, default=-1, 
                        help='If positive, allows validating on random subsets of the validation dataset during training.')
    parser.add_argument('--temperature', type=float, default=0.07, help='The temperature parameter for nll')

    add_management_args(parser)

    args = parser.parse_args()

    return args

class ContrastiveTransformations(object):
    """ helper class to prepare the data loading such that we sample 
    two different, random augmentations for each image in the batch"""
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

def info_nce_loss(outputs, t):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(outputs[:,None,:], outputs[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / t # temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        return nll, cos_sim, pos_mask
     
def evaluate_similarity(cos_sim, pos_mask):
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                                cos_sim.masked_fill(pos_mask, -9e15)],
                                dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        acc = (sim_argsort == 0).float().mean()
        acc_pos = 1+sim_argsort.float().mean()
        return acc, acc_pos

def validation(val_loader, model, t, num_samples):
        status = model.training
        model.eval()
        if num_samples >0: 
                # we select a subset of the validation dataset to validate on 
                # note: a different random sample is used every time
                random_indices = np.random.choice(range(len(val_loader.dataset)), size=num_samples, replace=False)
                _subset = Subset(val_loader.dataset, random_indices)
                val_loader =  DataLoader(_subset, 
                                        batch_size=val_loader.batch_size, 
                                        shuffle=False, num_workers=4, pin_memory=False)
        acc, acc_pos, total = 0.0, 0.0, 0.0
        for i,data in enumerate(val_loader):
                with torch.no_grad():
                        inputs, _ = data
                        inputs = torch.cat(inputs, dim=0)
                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        loss, C, M = info_nce_loss(outputs, args.temperature)
                        total += inputs.shape[0]
                        _acc, _acc_pos = evaluate_similarity(C,M)
                        acc += _acc
                        acc_pos += _acc_pos
                                          
        acc=(acc / total) * 100
        acc_poss=(acc_pos / total) * 100

        model.train(status)
        return acc.item(), acc_pos.item()


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
contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=256),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          normalize,
                                         ])

train_dataset = ImageFolder(imagenet_root+'train', ContrastiveTransformations(contrast_transforms, n_views=2))
val_dataset = ImageFolder(imagenet_root+'val', ContrastiveTransformations(contrast_transforms, n_views=2))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                        shuffle=False, num_workers=4, pin_memory=False)


print("Dataset ready.")

# initialising the model
model = resnet18(weights=None, num_classes=128*4)
# following https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
# we add an extra linear layer on top
model.fc = nn.Sequential(
            model.fc,  
            nn.ReLU(inplace=True),
            nn.Linear(4*128, 128)
        )

setproctitle.setproctitle('{}_{}_{}'.format("resnet18", "ssl", "imagenet"))

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

CHKPT_NAME = f'rn18-ssl-imagenet.ckpt' # obtaineed with seed = ?

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
        avg_acc, avg_acc_pos = 0.0, 0.0
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
                outputs = model(inputs)
                loss, C, M = info_nce_loss(outputs, args.temperature)
                loss.backward()
                optimizer.step()

                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, 'SSL', loss.item())
                avg_loss += loss
                total += inputs.shape[0]

                _acc, _acc_pos = evaluate_similarity(C,M)
                avg_acc += _acc.item()
                avg_acc_pos += _acc_pos.item()



        avg_acc  = (avg_acc/total)*100
        avg_acc_pos  = (avg_acc_pos/total)*100
        avg_loss = avg_loss/len(train_loader)

        if scheduler is not None:
                scheduler.step()
        
        val_acc, val_acc_pos = validation(val_loader, model, args.temperature, args.validate_subset)
        

        # best val accuracy -> selection bias on the validation set
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('\Train accuracy : {} %'.format(round(avg_acc, 2)), file=sys.stderr)
        print('\Train accuracy positive : {} %'.format(round(avg_acc_pos, 2)), file=sys.stderr)
        print('\Val accuracy : {} %'.format(round(val_acc, 2)), file=sys.stderr)
        print('\Val accuracy positive : {} %'.format(round(val_acc_pos, 2)), file=sys.stderr)
        
        df = {'epoch_loss':avg_loss,
        'epoch_train_acc':avg_acc,
        'epoch_train_acc_pos':avg_acc_pos,
        'epoch_val_acc':val_acc,
        'epoch_val_acc_pos':val_acc_pos}
        wandb.log(df)

        if args.checkpoints and (is_best or epoch==args.n_epochs-1):
                save_checkpoint({
                'epoch': args.n_epochs + 1,
                'state_dict': model.state_dict(),
                'best_acc': val_acc,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
                }, is_best, filename=CHKPT_NAME)
        
final_val_acc_D = val_acc

print("Training completed.")