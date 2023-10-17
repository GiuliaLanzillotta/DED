"""

Error note: when using the data from the script, need to divide the risk value by the batch size - because it was only divided by the number of batches
"""


import importlib
import json
import math
import os
import socket
import sys
import time
import numpy as np

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/utils')


import datetime
import uuid
from argparse import ArgumentParser

import torch

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageNet, ImageFolder
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar

from torch.nn.functional import one_hot, softmax


CHKPT_NAME = "rn50_2023-02-21_10-45-30_best.ckpt"
NUM_SAMPLES_F = 1000
NUM_SUBSET_SAMPLES = 100
SUBSET_P = 0.05 # encoding the percentage of data that (on average) is included in the subset 
DEVICE=[3] #NOTE fix this to whatever GPU you want to use
device = get_device(DEVICE)


def load_checkpoint(best=False, filename='checkpoint.pth.tar', distributed=False):
    path = base_path() + "chkpts" + "/" + "imagenet" + "/" + "resnet50/"
    if best: filepath = path + 'model_best.pth.tar'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath)
          if filename==CHKPT_NAME and not distributed: # modify Sidak's checkpoint
                new_state_dict = {k.replace('module.','',1):v for (k,v) in checkpoint['state_dict'].items()}
                checkpoint['state_dict'] = new_state_dict
          return checkpoint
    return None 

def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(-1.0, 1.0)
            m.bias.data.fill_(0)

# load the datasets ... 
imagenet_root = "/local/home/stuff/imagenet/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


train_transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
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

all_data = ConcatDataset([train_dataset, val_dataset])
all_data_loader = DataLoader(
        all_data, batch_size=256, shuffle=True,num_workers=4, pin_memory=True)

all_indices = set(range(len(all_data)))



# initialising the network: we need a teacher and many students 
teacher = resnet50(weights=None)
chkpt_name = f"checkpoint_90.pth.tar" #sidak's checkpoint
checkpoint = load_checkpoint(best=False, filename=CHKPT_NAME, distributed=False) #TODO: switch best off
teacher.load_state_dict(checkpoint['state_dict'])
teacher.to(device)
teacher.eval()
best_acc1 = checkpoint['best_acc1']


C = 1000

# storage of results
path = base_path() + "results" + "/" + "imagenet" + "/" + "resnet50" 
if not os.path.exists(path): os.makedirs(path)


for k in range(NUM_SAMPLES_F): 
    # initialising network k 
    #if k==2: break # for testing
    fnet = resnet50(weights=None)
    fnet.apply(weights_init_uniform)
    fnet.to(device)
    fnet.eval()

    print(f"Ready to go with network {k}!")

    print(f"Evaluating on all the data ...")
    risk = 0; total = 0
    progress_bar = ProgressBar(verbose=True)
    for i, data in enumerate(all_data_loader):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                labels = F.one_hot(labels, num_classes=C).to(torch.float)
                outputs_f = F.softmax(fnet(inputs),dim=1)
                risk += F.mse_loss(outputs_f, labels, reduction='sum')
                total+=labels.shape[0]

            progress_bar.prog(i, len(all_data_loader), -1, k, risk.item()/total )  

        # check type of risks variables ...
    if isinstance(risk, torch.Tensor): risk = risk.item()


    for s in range(NUM_SUBSET_SAMPLES): # each corresponds to a different dataset draw

        # computing the statistics 
        progress_bar = ProgressBar(verbose=True)
        e_risk=0; d_risk=0; total_subset = 0

        print(f"Random subset {s+1}/{NUM_SUBSET_SAMPLES} ...")

        random_indices = np.random.choice(list(all_indices), size=int(SUBSET_P*len(all_indices)), replace=False)
        subset = Subset(all_data, random_indices)
        subset_loader =  DataLoader(subset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

        for i, data in enumerate(subset_loader):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                labels = F.one_hot(labels, num_classes=C).to(torch.float)
                outputs_f = F.softmax(fnet(inputs),dim=1)
                outputs_t = F.softmax(teacher(inputs),dim=1)
                e_risk+=F.mse_loss(outputs_f, labels, reduction='sum')
                d_risk+=F.mse_loss(outputs_f, outputs_t, reduction='sum')
                total_subset+=labels.shape[0]

            progress_bar.prog(i, len(subset_loader), s, k, (e_risk-d_risk).item()/total_subset)  

        # check type of risks variables ...
        if isinstance(e_risk, torch.Tensor): e_risk = e_risk.item()
        if isinstance(d_risk, torch.Tensor): d_risk = d_risk.item()

        risk_log = {}
        risk_log['risk'] = risk/len(all_data)
        risk_log['e_risk'] = e_risk/total_subset
        risk_log['d_risk'] = d_risk/total_subset
        risk_log['subset_size'] = total_subset
        risk_log['subset_fraction'] = SUBSET_P
        risk_log['s'] = s
        risk_log['k'] = k
 
        with open(path+ "/RISK.txt", 'a') as f:
                f.write(json.dumps(risk_log) + '\n')