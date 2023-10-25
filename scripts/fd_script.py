"""
Script to evaluate the functional distance of trained students 
to the teacher and labels. 


#TODO: re-do with hellinger distance for better interpretability https://en.wikipedia.org/wiki/Hellinger_distance 

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
from utils.eval import evaluate, validation_and_agreement, distance_models, evaluate_regression, hellinger


# Evaluating the distribution of |f(x)-t(x)| 
from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar
from torch.utils.data import ConcatDataset
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18, googlenet, efficientnet_b0, mobilenet_v3_large

from datasets.data_utils import load_dataset


DEVICE=[4] #NOTE 
SEEDS = [11, 13, 21, 33, 55]
BUFFER_SIZES = [600000, 120000, 60000]
ALPHA = [1.0,0.0]
TRAIN_SUBSET = 50000 # cifar10 length
DATASET = 'cifar10'
MODEL = 'mnet' #mnet
C = 10 # number of classes
is_train = True

device = get_device(DEVICE)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(d) for d in DEVICE])
if DATASET=='imagenet':  
        CHKPT_NAME = "rn50_2023-02-21_10-45-30_best.ckpt" # CHKPT_NAME = f'mnet-teacher.ckpt'
        CHKPT_PATH = base_path() + "chkpts" + "/" + "imagenet" + "/" + "resnet50/"  
        LOGS_PATH = base_path() + "results" + "/" + "imagenet" + "/" + "resnet50/"
elif DATASET=='cifar10':
       CHKPT_NAME = f'mnet-teacher.ckpt'
       CHKPT_PATH = base_path() + "chkpts" + "/" + "cifar5m" + "/" + "mnet/" 
       LOGS_PATH = base_path() + "results" + "/" + "cifar5m" + "/" + "mnet/"   

def load_checkpoint(device, best=False, filename='checkpoint.pth.tar'):
    path = CHKPT_PATH
    if best: filepath = path + 'model_best.pth.tar'
    else: filepath = path + filename
    print(f"Attempt to load checkpoint {filepath} ...")
    if os.path.exists(filepath):
          print("POSITIVE - file exists")
          checkpoint = torch.load(filepath, map_location=device)
          if filename==CHKPT_NAME and  DATASET=='imagenet': # modify Sidak's checkpoint
                new_state_dict = {k.replace('module.','',1):v for (k,v) in checkpoint['state_dict'].items()}
                checkpoint['state_dict'] = new_state_dict
          return checkpoint
    return None 


train_dataset, val_dataset = load_dataset(DATASET, augment=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=False)

all_indices = set(range(len(train_dataset)))
random_indices = np.random.choice(list(all_indices), size=TRAIN_SUBSET, replace=False)
train_subset = Subset(train_dataset, random_indices)
train_subset_loader =  DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)


# loading the teacher
if MODEL=='rn50': # resnet 50 
        teacher = resnet50(weights=None)
elif MODEL == 'mnet': # mobilenet
       teacher = mobilenet_v3_large(num_classes=10) # adjusting for CIFAR 

checkpoint = load_checkpoint(device, best=False, filename=CHKPT_NAME) 
teacher.load_state_dict(checkpoint['state_dict'])
teacher.to(device)
teacher.eval()

loader = train_subset_loader if is_train else val_loader
filename = "FDIST_" +DATASET+ "_test.txt" if not is_train else "FDIST_"+DATASET+".txt"
count=0
total = len(SEEDS)*4
for buffer_size in BUFFER_SIZES:
    for seed in SEEDS: 
            for alpha in ALPHA:  
                    count+=1
                    

                    start = time.time()
                    
                    STUD_CHKPT = f'{MODEL}-student-{seed}-{buffer_size}-{alpha}.ckpt'

                    # loading the student
                    if MODEL=='rn50': # resnet 50 
                        student = resnet50(weights=None)
                    elif MODEL == 'mnet': # mobilenet
                        student = mobilenet_v3_large(num_classes=10) # adjusting for CIFAR 

                    checkpoint = load_checkpoint(device, best=False, filename=STUD_CHKPT) 
                    student.load_state_dict(checkpoint['state_dict'])
                    student.to(device)
                    student.eval()

                    progress_bar = ProgressBar(verbose=True)
                    for i,data in enumerate(loader):
                            with torch.no_grad():
                                    inputs, labels = data
                                    inputs, labels = inputs.to(device), labels.to(device)
                                    labels = F.one_hot(labels, num_classes=C).to(torch.float)
                                    outputs_s = F.softmax(student(inputs),dim=1)
                                    outputs_t = F.softmax(teacher(inputs),dim=1)
                                    distance_t = hellinger(outputs_s,outputs_t).tolist()
                                    distance_l = hellinger(outputs_s,labels).tolist()
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
                                    with open(LOGS_PATH+ filename, 'a') as f:
                                            f.write(json.dumps(dict) + '\n')
                                    
                                    dict['teacher'] = False
                                    dict['distance'] = d_l
                                    with open(LOGS_PATH+ filename, 'a') as f:
                                            f.write(json.dumps(dict) + '\n')
                        
                    end = time.time()
                    print(f"Took {end-start} s to complete iteration {count}. {total-count} to go.")
