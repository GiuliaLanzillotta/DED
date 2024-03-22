""" Script to compute cross-cka matrices for CIFAR10"""


import importlib
import json
import math
import os
import socket
import sys
import time
from tqdm import tqdm



internal_path = os.path.abspath(os.path.join('.'))
sys.path.append(internal_path)
sys.path.append(internal_path + '/datasets')
sys.path.append(internal_path + '/utils')

import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch
import numpy as np
import pandas as pd 
import json

from copy import deepcopy

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib.collections import LineCollection



import shutil
from utils.args import add_management_args, add_rehearsal_args
from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar
from utils.stil_losses import *
from utils.nets import *
from utils.kernels import *
from torchvision.datasets import ImageNet, ImageFolder
from torch.utils.data import ConcatDataset
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, googlenet, efficientnet_b0, mobilenet_v3_large
from utils.eval import evaluate, validation_and_agreement, distance_models, validation_agreement_function_distance
from dataset_utils.data_utils import load_dataset, CIFAR100sparse2coarse

def load_checkpoint(best=False, filename='checkpoint.pth.tar', type='mnet', device="cpu"):
    """ Available network types: [mnet, convnet]"""
    path = base_path() + "chkpts" + "/" + "cifar5m" + "/" + f"{type}/"
    if best: filepath = path + 'model_best.ckpt'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath, map_location=device)
          return checkpoint
    return None 


GPUID = 0
SEEDS = [11, 13, 33]
BUFFER_SIZES = [1200, 6000, 12000, 24000, 48000, 60000, 90000, 120000]
TEMPERATURES = [0.1, 1.0, 3.0, 5.0, 10.0, 20.0, 100.0, 10000.0]
NUM_BATCHES = 100 #approx 1K samples
NUM_SAMPLES = 500000

device = get_device([GPUID])

C5m_train, C5m_test = load_dataset('cifar5m', augment=False)

print(f"Randomly drawing {NUM_SAMPLES} samples for the Cifar5M base")
all_indices = set(range(len(C5m_train)))
random_indices = np.random.choice(list(all_indices), size=NUM_SAMPLES, replace=False)
data = Subset(C5m_train, random_indices)
# shuffling the validation data
random_indices = np.random.choice(list(range(len(C5m_test))), size=len(C5m_test), replace=False)
valdata = Subset(C5m_test, random_indices)
loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=4, pin_memory=False)
valloader = DataLoader(valdata, batch_size=128, shuffle=False, num_workers=4, pin_memory=False)

basenet = CNN(num_classes=10, c=20) # adjusting for CIFAR 
teacher = deepcopy(basenet)
CHKPT_NAME = f'CNN-teacher.ckpt' 
checkpoint = load_checkpoint(best=False, filename=CHKPT_NAME, type='CNN', device=device) 
if checkpoint: 
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.to(device)
teacher.eval();


KT = compute_empirical_kernel(valloader, teacher, device=device, num_batches=NUM_BATCHES)

for T in TEMPERATURES:
    for b in BUFFER_SIZES:
        kernels = []
        total = 0
        for alpha in [0.0,1.0]:
                for seed in SEEDS:
                        student = deepcopy(basenet)
                        if alpha==0.0:
                                STUDNT_NAME = f'CNN-student-{seed}-{b}-{alpha}-{T}.ckpt'
                        else: 
                                STUDNT_NAME = f'CNN-student-{seed}-{b}-{alpha}-1.0.ckpt'
                        print(STUDNT_NAME)
                        checkpoint = load_checkpoint(best=False, filename=STUDNT_NAME, type='CNN', device=device) 
                        if checkpoint: 
                                student.load_state_dict(checkpoint['state_dict'])
                                student.to(device)
                                student.eval();
                                KS = compute_empirical_kernel(valloader, student, device=device, num_batches=NUM_BATCHES)
                                kernels.append(KS)
                                total+=1
        print(f"{total} networks evaluated for T={T} and b={b}")

        results = np.zeros((6,6))
        for i, K in tqdm(enumerate(kernels)):
            for j,Q in enumerate(kernels):
                results[i,j] = centered_kernal_alignment(K,Q).item()
        path = f"./logs/results/cifar5m/CNN/cross-cka-{T}-{b}.npy"
        print(f"Saving {path}")
        np.save(path, results)        
        
