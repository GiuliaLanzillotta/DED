
import importlib
import json
import math
import os
import socket
import sys
import time

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')
sys.path.append(mammoth_path + '/utils')


import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch

from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageNet, ImageFolder
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from datasets import NAMES as DATASET_NAMES
from datasets import ContinualDataset, get_dataset
from models import get_all_models, get_model


import shutil
from utils.args import add_management_args, add_rehearsal_args
from utils.conf import set_random_seed, get_device
from utils.loggers import *
from utils.status import ProgressBar
from utils.buffer import Buffer

from torch.nn.functional import one_hot, softmax



NUM_SAMPLES_F = 100
DEVICE=[0] #NOTE fix this to whatever GPU you want to use
device = get_device(DEVICE)


def load_checkpoint(best=False, filename='checkpoint.pth.tar', distributed=False):
    path = base_path() + "/chkpts" + "/" + "imagenet" + "/" + "resnet50/"
    if best: filepath = path + 'model_best.pth.tar'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath)
          if filename=='checkpoint_90.pth.tar' and not distributed: # modify Sidak's checkpoint
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
        all_data, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)



# initialising the network: we need a teacher and many students 
teacher = resnet50(weights=None)
chkpt_name = f"checkpoint_90.pth.tar" #sidak's checkpoint
checkpoint = load_checkpoint(best=False, filename=chkpt_name, distributed=False) #TODO: switch best off
teacher.load_state_dict(checkpoint['state_dict'])
teacher.to(device)
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

    # computing the statistics 
    progress_bar = ProgressBar(verbose=True)

    sum1_delta = 0; sum2_delta = 0; sum1_ce = 0; sum2_ce = 0; total = 0

    print(f"Ready to go with network {k}!")

    for i, data in enumerate(all_data_loader):
        #if i==10: break # for testing
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs_f = fnet(inputs)
            outputs_t = teacher(inputs)
            delta = softmax(outputs_t.view(-1,C)) - one_hot(labels, num_classes=C).view(-1,C)
            l_delta = F.cross_entropy(outputs_f, delta)
            l_ce = F.cross_entropy(outputs_f, labels)

            sum1_delta += l_delta.sum()
            sum2_delta += (l_delta**2).sum()
            sum1_ce += l_ce.sum()
            sum2_ce += (l_ce**2).sum()

            total += labels.shape[0]

            running_delta = (sum2_delta/total - (sum1_delta/total)**2)*(1/(total-1)) + (sum1_delta/total)**2
            running_ce = (sum2_ce/total - (sum1_ce/total)**2)*(1/(total-1))
            
        progress_bar.prog(i, len(all_data_loader), running_delta.item(), 'D', running_ce.item())  


    delta_var = (sum2_delta/total - (sum1_delta/total)**2)*(total/(total-1))
    ce_var = (sum2_ce/total - (sum1_ce/total)**2)*(total/(total-1))

    inequality_log = {}
    inequality_log['deltaV'] = delta_var.item()
    inequality_log['deltaM'] = (sum1_delta/total).item()
    inequality_log['ceV'] = ce_var.item()

    with open(path+ "/LVI.txt", 'a') as f:
            f.write(json.dumps(inequality_log) + '\n')