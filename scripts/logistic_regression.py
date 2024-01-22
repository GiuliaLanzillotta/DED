# Giulia Lanzillotta . 17.10.2023
# Logistic regression training experiment script 

"""
Here we take any pretrained student network and train a logistic regression classifier on top of its features.  

Example command: 
python scripts/logistic_regression.py --dataset cifar5m --network CNN --seed 11 --temperature 20 --alpha 0 --buffer_size 12000 --gpus_id 4 --batch_size 256  

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
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, googlenet, efficientnet_b0, mobilenet_v3_large
import torchvision.transforms as transforms
from torch.nn.utils import parameters_to_vector
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

import shutil
from utils.args import add_management_args, add_rehearsal_args
from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar
from utils.stil_losses import *
from utils.nets import *
from utils.optim import *
from utils.eval import evaluate, validation_and_agreement, distance_models, validation_agreement_function_distance, evaluate_CKA_teacher
from dataset_utils.data_utils import load_dataset

try:
    import wandb
except ImportError:
    wandb = None

NUM_CLASSES = {
        "cifar10":10,
        "cifar5m":10,
        "cifar100":100,
        "imagenet":1000
}

MAX_SAMPLES = 100000 #TODO: implement - only necessary for imagenet experiments at the moment.

def load_checkpoint(device, best=False, filename='checkpoint.pth.tar', dataset="cifar100", network="resnet18"):
    path = base_path() + "chkpts" + "/" + f"{dataset}" + "/" + f"{network}/"
    if best: filepath = path + 'model_best.ckpt'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath, map_location=device)
          return checkpoint
    return None 

def parse_args():
    torch.set_num_threads(4)
    parser = ArgumentParser(description='script-experiment', allow_abbrev=False)
    parser.add_argument('--temperature', type=float, default=1., help='Temperature (prop to entropy) of the teacher outputs - only used with KL.')
    parser.add_argument('--buffer_size', type=int, required=True, help='The size of the memory buffer.')
    parser.add_argument('--alpha', type=float, default=0.5, required=True, help='The weight of labels vs logits in the distillation loss (when alpha=1 only true labels are used)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed.')
    parser.add_argument('--batch_size', type=int, default = 256, help='Batch size.')

    parser.add_argument('--dataset', type=str, required=True, choices=["cifar100","imagenet","cifar5m"],help='Which dataset to use.') 
    parser.add_argument('--network', type=str, required=True, choices=["resnet18","CNN","resnet50"],help='Which network to use.') 

    parser.add_argument('--gpus_id', nargs="+", type=int, default=[],help='GPU devices identifier. Empty list means CPU only.')
    
    args = parser.parse_args()
    return args


args = parse_args()
print(args)
experiment_log = vars(args)

if args.seed is not None:
        set_random_seed(args.seed)


print(file=sys.stderr)

DATASET = args.dataset
NETWORK = args.network
# dataset -> cifar100 for the teacher and cifar5m for the student
train, val = load_dataset(DATASET, augment=False)
if DATASET=="cifar5m":
    print(f"Randomly drawing {60000} samples from CIFAR5m")
    all_indices = set(range(len(train)))
    random_indices = np.random.choice(list(all_indices), size=60000, replace=False)
    train = Subset(train, random_indices)
    
train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)


# start the training 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(d) for d in args.gpus_id])

device = get_device([0]) # returns the first device in the list
print(f"Selected device {device}")
progress_bar = ProgressBar(verbose=True)



print("Loading the student ... ")
start = time.time()
# re-initialise model 
if NETWORK=="resnet18": student = resnet18(num_classes=NUM_CLASSES[DATASET])
elif NETWORK=="CNN": student = CNN(c=20, num_classes=NUM_CLASSES[DATASET], use_batch_norm=True)
elif NETWORK=="resnet50": student = resnet50(weights=None, num_classes=NUM_CLASSES[DATASET])
student = feature_wrapper(student) # add 'get_features' function
student.to(device)

chkpt_filename=f'{NETWORK}-student-{args.seed}-{args.buffer_size}-{args.alpha}-{args.temperature}.ckpt'
checkpoint = load_checkpoint(device=device, best=False, filename=chkpt_filename, dataset=DATASET, network=NETWORK) 
assert checkpoint is not None, "student checkpoint not found."
print("Student checkpoint found. ")
student.load_state_dict(checkpoint['state_dict'])
student.to(device)
start_acc = checkpoint['best_acc']    
student.eval()

# check the initial val accuracy of the student and log it
experiment_log['final_val_acc_S'] = evaluate(student, val_loader, device)

# collecting features to build the data matrix X 
X = []
Y = []
total = 0
for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad(): phi_s = student.get_features(inputs)
                X.append(phi_s.cpu().numpy())
                Y.append(labels.cpu().numpy())
                       
                       
                total += labels.shape[0]
                progress_bar.prog(i, len(train_loader), i/len(train_loader), 'Collecting training data',  0)
        
X = np.vstack(X).reshape((total, phi_s.size(1)))
Y = np.hstack(Y).reshape((total,))

#print("Scaling the features.")
#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)

# Using sklearn logistic regression 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
print("Fitting logistic model...")
clf = LogisticRegression(penalty='l2', C=1., fit_intercept=True, random_state=args.seed, max_iter=100).fit(X, Y)
print("Linear retraining completed. Full evaluation and logging...")
train_acc = clf.score(X,Y)
print(f"Train accuracy: {train_acc}")
Xt = []
Yt = []
total = 0
for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad(): phi_s = student.get_features(inputs)
                Xt.append(phi_s.cpu().numpy())
                Yt.append(labels.cpu().numpy())
                       
                       
                total += labels.shape[0]
                progress_bar.prog(i, len(val_loader), i/len(val_loader), 'Collecting test data',  0)

         
Xt = np.vstack(Xt).reshape((total, phi_s.size(1)))
Yt = np.hstack(Yt).reshape((total,))

#Xt = scaler.transform(Xt)
val_acc = clf.score(Xt, Yt)
end = time.time()

print(f"Val accuracy: {val_acc}")
experiment_log['buffer_train_time'] = end-start
experiment_log['final_train_acc_S'] = train_acc
experiment_log['final_val_acc_S'] = val_acc

# dumping everything into a log file
path = base_path() + "results" + "/" + f"{DATASET}" + "/" + f"{NETWORK}" 
if not os.path.exists(path): os.makedirs(path)
with open(path+ "/logs_logistic_regression.txt", 'a') as f:
        f.write(json.dumps(experiment_log) + '\n')