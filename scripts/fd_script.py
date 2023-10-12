"""
Script to evaluate the functional distance of trained students 
to the teacher and labels. 


notes:we evaluated 30k-alpha=0 so far 

"""
import json
import math
import os
import socket
import sys
import time
import datetime
import uuid
from argparse import ArgumentParser
import setproctitle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib.collections import LineCollection
import json

internal_path = os.path.abspath(os.path.join('.'))
sys.path.append(internal_path)
sys.path.append(internal_path + '/datasets')
sys.path.append(internal_path + '/utils')

import shutil
from utils.args import add_management_args, add_rehearsal_args
from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar
from utils.stil_losses import *
from utils.nets import *
from utils.eval import evaluate, validation_and_agreement, distance_models, evaluate_regression


# Evaluating the distribution of |f(x)-t(x)| 
from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar
from torch.utils.data import ConcatDataset

DEVICE=[5] #NOTE 
CHKPT_NAME = "rn50_2023-02-21_10-45-30_best.ckpt"
SEEDS = [21,33]
BUFFER_SIZES = [30000,90000]
ALPHA = [0.0,1.0]

device = get_device(DEVICE)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(d) for d in DEVICE])
path = base_path() + "results" + "/" + "imagenet" + "/" + "resnet50" +"/"

def load_checkpoint(device, best=False, filename='checkpoint.pth.tar', distributed=False):
    path = base_path() + "chkpts" + "/" + "imagenet" + "/" + "resnet50/"
    if best: filepath = path + 'model_best.pth.tar'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath, map_location=device)
          if filename==CHKPT_NAME and not distributed: # modify Sidak's checkpoint
                new_state_dict = {k.replace('module.','',1):v for (k,v) in checkpoint['state_dict'].items()}
                checkpoint['state_dict'] = new_state_dict
          return checkpoint
    return None 

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
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=False)

# loading the teacher
teacher = resnet50(weights=None)
checkpoint = load_checkpoint(device, best=False, filename=CHKPT_NAME, distributed=False) 
teacher.load_state_dict(checkpoint['state_dict'])
teacher.to(device)
teacher.eval()



is_train = False
loader = train_loader if is_train else val_loader
filename = "FDIST" + "_test.txt" if not is_train else ".txt"
count=0
total = len(SEEDS)*4
for buffer_size in BUFFER_SIZES:
    for seed in SEEDS: 
            for alpha in ALPHA:  
                    count+=1
                    

                    start = time.time()
                    
                    STUD_CHKPT = f'rn50-student-{seed}-{buffer_size}-{alpha}.ckpt'

                    # loading the student
                    student = resnet50(weights=None)
                    checkpoint = load_checkpoint(device, best=False, filename=STUD_CHKPT, distributed=False) 
                    student.load_state_dict(checkpoint['state_dict'])
                    student.to(device)
                    student.eval()

                    progress_bar = ProgressBar(verbose=True)
                    for i,data in enumerate(loader):
                            with torch.no_grad():
                                    inputs, labels = data
                                    inputs, labels = inputs.to(device), labels.to(device)
                                    labels = F.one_hot(labels, num_classes=1000).to(torch.float)
                                    outputs_s = F.softmax(student(inputs),dim=1)
                                    outputs_t = F.softmax(teacher(inputs),dim=1)
                                    distance_t = torch.norm(outputs_s-outputs_t, dim=1, p=2).tolist()
                                    distance_l = torch.norm(outputs_s-labels, dim=1, p=2).tolist()
                            progress_bar.prog(i, len(loader), count, 'train data' if is_train else 'val data', distance_t[0])
                            
                            distance_log = {}
                            distance_log['alpha'] = alpha
                            distance_log['batch'] = i
                            distance_log['buffer_size'] = buffer_size
                            distance_log['seed'] = seed 
                            distance_log['test'] = not is_train

                            for d_t, d_l in zip(distance_t, distance_l): 
                                    dict = distance_log
                                    dict['teacher'] = True
                                    dict['distance'] = d_t
                                    with open(path+ filename, 'a') as f:
                                            f.write(json.dumps(dict) + '\n')
                                    
                                    dict['teacher'] = False
                                    dict['distance'] = d_l
                                    with open(path+ filename, 'a') as f:
                                            f.write(json.dumps(dict) + '\n')
                        
                    end = time.time()
                    print(f"Took {end-start} s to complete iteration {count}. {total-count} to go.")
