# Giulia Lanzillotta . 17.10.2023
# Cifar100 training experiment script 

"""

The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class.


example commands: 

python scripts/cifar100_linear_retraining.py  --seed 11 --alpha 0 --temperature 10 --gpus_id 6 --buffer_size 12000 --distillation_type vanilla --batch_size 128  --checkpoints --notes cifar100-resnet18-linretraining --wandb_project DataEfficientDistillation


Teacher recipe: https://huggingface.co/edadaltocg/resnet18_cifar100

We here load pre-trained student models and retrain only the linear head on the full dataset. 

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

LOGITS_MAGNITUDE_TEACHER = 1.0 
AUGMENT = False
SIZE = 150

# For teacher training we take inspiration from these results
# https://github.com/weiaicunzai/pytorch-cifar100 
def setup_optimizerNscheduler(args, model):
        """Iteer_per_epoch is a necessary argument for the teacher optimizer"""
        epochs = args.n_epochs
        if not args.optim_adam:
                optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
                                lr=args.lr, 
                                weight_decay=args.optim_wd, 
                                momentum=args.optim_mom,
                                nesterov=args.optim_nesterov)
        else: 
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], 
                                             lr = args.lr, 
                                             weight_decay=args.optim_wd)
                
        if not args.optim_cosineanneal: 
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        else: 
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-args.optim_warmup)

        if args.optim_warmup > 0: # initialise warmup scheduler
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                                     start_factor=0.01, 
                                                                     total_iters=args.optim_warmup)
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
                                                                schedulers=[warmup_scheduler, scheduler], 
                                                                milestones=[args.optim_warmup])
        return optimizer, scheduler

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    path = base_path() + "/chkpts" + "/" + "cifar100" + "/" + "resnet18/"
    if not os.path.exists(path): os.makedirs(path)
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(path+filename, path+'model_best.ckpt')

def load_checkpoint(best=False, filename='checkpoint.pth.tar', distributed=False):
    path = base_path() + "chkpts" + "/" + "cifar100" + "/" + "resnet18/"
    if best: filepath = path + 'model_best.ckpt'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath)
          return checkpoint
    return None 

def parse_args():
    torch.set_num_threads(4)
    parser = ArgumentParser(description='script-experiment', allow_abbrev=False)
    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--checkpoints', action='store_true', help='Storing a checkpoint at every epoch. Loads a checkpoint if present.')
    parser.add_argument('--checkpoints_stud', action='store_true', help='Storing a checkpoint for the student.')
    parser.add_argument('--pretrained', action='store_true', help='Using a pre-trained network instead of training one.')
    parser.add_argument('--optim_wd', type=float, default=0, help='optimizer weight decay.')
    parser.add_argument('--optim_adam', default=False, action='store_true', help='Using the Adam optimizer instead of SGD.')
    parser.add_argument('--optim_mom', type=float, default=0, help='optimizer momentum.')
    parser.add_argument('--optim_warmup', type=int, default=0, help='Number of warmup epochs.')
    parser.add_argument('--optim_nesterov', default=False, action='store_true', help='optimizer nesterov momentum.')
    parser.add_argument('--optim_cosineanneal', default=False, action='store_true', help='Enabling cosine annealing of learning rate..')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default = 256, help='Batch size.')
    parser.add_argument('--validate_subset', type=int, default=-1, 
                        help='If positive, allows validating on random subsets of the validation dataset during training.')
    parser.add_argument('--temperature', type=float, default=1., help='Temperature (prop to entropy) of the teacher outputs - only used with KL.')

    add_management_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--alpha', type=float, default=0.5, required=False,
                        help='The weight of labels vs logits in the distillation loss (when alpha=1 only true labels are used)')
    parser.add_argument('--conditional_teacher', default=False, action='store_true',
                        help='If provided, the teacher use is conditioned on its performance.')
    parser.add_argument('--MSE', default=False, action='store_true',
                        help='If provided, the MSE loss is used for the student with labels .')
    parser.add_argument('--teacher_targets', default=False, action='store_true',
                        help='If provided, using teacher targets instead of one-hot labels.')
    parser.add_argument('--distillation_type', type=str, default='vanilla', choices=['vanilla', 'topK', 'inner', 'inner-parallel', 'topbottomK','randomK'],
                        help='Selects the distillation type, which determines the distillation loss.')
    parser.add_argument('--gamma', type=float, default=1.0, help='The mixing weight for mixed inner distillation')
    args = parser.parse_args()

    return args


args = parse_args()
experiment_log = vars(args)
# Add uuid, timestamp and hostname for logging
args.conf_jobnum = str(uuid.uuid4())
args.conf_timestamp = str(datetime.datetime.now())
args.conf_host = socket.gethostname()

if args.seed is not None:
        set_random_seed(args.seed)

# dataset -> cifar100 for the teacher and cifar5m for the student
C100_train, C100_val = load_dataset('cifar100', augment=AUGMENT)

# initialising the model
teacher = resnet18(num_classes=100)
params = count_parameters(teacher)
print(f"Teacher created with {params} parameters")

setproctitle.setproctitle('{}_{}_{}'.format(f"resnet18", args.buffer_size if 'buffer_size' in args else 0, "cifar100"))

# start the training 
print(args)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(d) for d in args.gpus_id])
if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if args.wandb_name is None: 
                name = str.join("-",["cifar100-linear", f"resnet18", args.conf_timestamp])
        else: name = args.wandb_name
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
                        name=name, notes=args.notes, config=vars(args)) 
        args.wandb_url = wandb.run.get_url()
device = get_device([0]) # returns the first device in the list
if args.distributed=='dp': 
      print(f"Parallelising training on {len(args.gpus_id)} GPUs.") 
      teacher = torch.nn.DataParallel(teacher, device_ids=args.gpus_id).cuda()
teacher.to(device)
progress_bar = ProgressBar(verbose=not args.non_verbose)


print(file=sys.stderr)
train_loader = DataLoader(C100_train, batch_size=args.batch_size, 
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(C100_val, batch_size=args.batch_size, 
                        shuffle=False, num_workers=4, pin_memory=False)


CHKPT_NAME = f'resnet18-teacher.ckpt' # obtaineed with seed = 11
checkpoint = load_checkpoint(best=False, filename=CHKPT_NAME, distributed=not args.distributed=='no') 
if checkpoint: 
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.to(device)
        start_epoch = checkpoint['epoch']
        final_val_acc_D = checkpoint['best_acc']

df = {'final_val_acc_D':final_val_acc_D}
wandb.log(df)
experiment_log['final_val_acc_D'] = final_val_acc_D

teacher.eval() # set the main model to evaluation


print("Starting student training ... ")
start = time.time()
# re-initialise model 
student = resnet18(num_classes=100)
student = feature_wrapper(student) # add 'get_features' function
student.to(device)

chkpt_filename=f'resnet18-student-{args.seed}-{args.buffer_size}-{args.alpha}-{args.temperature}.ckpt'
checkpoint = load_checkpoint(best=False, filename=chkpt_filename, distributed=False) 
if checkpoint: 
        print("Student checkpoint found. ")
        student.load_state_dict(checkpoint['state_dict'])
        student.to(device)
        start_acc = checkpoint['best_acc']     
else: raise Exception("Student checkpoint not found, you must provide an existing student checkpoint for linear retraining.") 

student.eval()

for (n,m) in student.named_children():
       if n != "fc":
              print("Freezing ",n)
              for p in m.parameters():
                     p.requires_grad = False
       else: 
              print("Resetting ",n)
              m.reset_parameters()
       

optimizer, scheduler = setup_optimizerNscheduler(args, student)


average_magnitude=0
alpha = args.alpha
T = args.temperature
for e in range(args.n_epochs):
        if args.debug_mode and e > 3: # only 3 batches in debug mode
                break
        avg_loss = 0.0
        best_acc = 0.0
        correct, total, agreement, function_distance = 0.0, 0.0, 0.0, 0.0
        for i, data in enumerate(train_loader):
                if args.debug_mode and i>10:
                       break
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad(): 
                       logits = teacher(inputs)
                       phi_s = student.get_features(inputs)
                       
                outputs = student.forward_head(phi_s)

                if LOGITS_MAGNITUDE_TEACHER == 1: # estimate during the first epoch
                       average_non_max = (logits.sum(dim=1) - logits.max(dim=1)[0])/9 # average over the non-max outputs
                       average_magnitude += (logits.max(dim=1)[0] - average_non_max).sum(dim=0) 
                
                if args.MSE: 
                       labels_loss = F.mse_loss(outputs, F.one_hot(labels, num_classes=100).to(torch.float) * LOGITS_MAGNITUDE_TEACHER, reduction='none').mean(dim=1)  # Bobby's correction
                       logits_loss = F.mse_loss(outputs, logits, reduction='none').mean(dim=1)
                else:
                      labels_loss = F.cross_entropy(outputs, labels, reduction='none')
                      logits_loss = F.kl_div(input=F.log_softmax(outputs/T, dim=1), target=F.softmax(logits/T, dim=1), log_target=False, reduction='none').sum(dim=1) * (T**2) # temperature rescaling (for gradients)
                
                if args.teacher_targets: 
                        loss = logits_loss.mean()
                else: loss =labels_loss.mean()


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                assert not math.isnan(loss)


                _, pred = torch.max(outputs.data, 1)
                _, pred_t = torch.max(logits.data, 1)
                correct += torch.sum(pred == labels).item()
                agreement += torch.sum(pred == pred_t).item()
                total += labels.shape[0]

                progress_bar.prog(i, len(train_loader), e, 'Student', loss.item())
                avg_loss += loss*(labels.shape[0])
        
        avg_loss = avg_loss/total
        if scheduler is not None:
                scheduler.step()
        
        if LOGITS_MAGNITUDE_TEACHER==1: 
               average_magnitude = average_magnitude/total
               LOGITS_MAGNITUDE_TEACHER = average_magnitude
               print(f"Setting LMT to {LOGITS_MAGNITUDE_TEACHER}")
        
        train_acc = (correct/total) * 100
        train_agreement = (agreement/total) * 100      
        # measure distance in parameter space between the teacher and student models 
        val_acc, val_agreement = validation_and_agreement(student, teacher, val_loader, device, num_samples=args.validate_subset)
        

        print('\nTrain accuracy : {} %'.format(round(train_acc, 2)), file=sys.stderr)
        print('\Val accuracy : {} %'.format(round(val_acc, 2)), file=sys.stderr)
        
        df = {'epoch_loss_S':avg_loss,
              'epoch_train_acc_S':train_acc,
              'epoch_train_agreement':train_agreement,
              'epoch_val_acc_S':val_acc,
              'epoch_val_agreement':val_agreement}

        wandb.log(df)
        
print("Linear retraining completed. Full evaluation and logging...")
end = time.time()

experiment_log['buffer_train_time'] = end-start
experiment_log['final_train_acc_S'] = train_acc
experiment_log['network'] = f'resnet18'
val_acc, val_agreement, val_function_distance = validation_agreement_function_distance(student, teacher, val_loader, device)
 
experiment_log['final_val_acc_S'] = val_acc
experiment_log['final_train_agreement'] = train_agreement
experiment_log['final_val_agreement'] = val_agreement
experiment_log['final_val_function_distance'] = val_function_distance

if not args.nowand:
        wandb.finish()

# dumping everything into a log file
path = base_path() + "results" + "/" + "cifar100" + "/" + f"resnet18" 
if not os.path.exists(path): os.makedirs(path)
with open(path+ "/logs_linear_retraining.txt", 'a') as f:
        f.write(json.dumps(experiment_log) + '\n')