# Giulia Lanzillotta . 17.10.2023
# Cifar 5M training experiment script 

"""

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
We use cifar5m, an extension to 5 mln images in order to train the student on more images than the teacher. 

We use a teacher trained with self-supervision on other image, then learn a head for the current dataset and apply distillation. 


example commands: 

python scripts/cifar5m_SSL.py --seed 11 --alpha 0 --gpus_id 1 --buffer_size 20000 --distillation_type vanilla --batch_size 128  --checkpoints --notes cifar5msmall-distillation --wandb_project DataEfficientDistillation


Using hyperparameters from Torch recipe https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning 

Mobilenets: compact, no-residuals CNNs https://arxiv.org/pdf/1704.04861.pdf 

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
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18, googlenet, efficientnet_b0, \
        mobilenet_v3_large, resnext101_64x4d
import torchvision.transforms as transforms
from torch.nn.utils import parameters_to_vector


import shutil
from utils.args import add_management_args, add_rehearsal_args
from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar
from utils.stil_losses import *
from utils.nets import *
from utils.eval import evaluate, validation_and_agreement, validation_agreement_function_distance
from dataset_utils.data_utils import load_dataset

try:
    import wandb
except ImportError:
    wandb = None

LOGITS_MAGNITUDE_TEACHER = 1.0 #TODO
AUGMENT = True
TEACHER_DATA_SIZE = 1000000
CHKPT_NAME = f'rn50-ssl-cifar10.ckpt' # obtaineed with seed = 11


class TeacherNet(nn.Module):
    # https://github.com/leftthomas/SimCLR/blob/master/model.py
    def __init__(self, feature_dim=128):
        super(TeacherNet, self).__init__()

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


class StudentNet(nn.Module):
    # a mobilenet with a big representation layer
    def __init__(self):
        super(StudentNet, self).__init__()

        net = mobilenet_v3_large(num_classes=10) # adjusting for CIFAR 

        # encoder
        self.f = net.features
        
        self.rescale = nn.Sequential(nn.Linear(960, 2048, bias=False), nn.ReLU(inplace=True))

        
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 10, bias=True))

    def freeze_backbone(self):
        # Freeze features parameters
        print("Freezing backbone ... ")
        for param in self.f.parameters():
                param.requires_grad = False
        for param in self.rescale.parameters():
                param.requires_grad = False
        
    def features(self, x):
        x = self.f(x)
        x = torch.flatten(x, 1)
        x = self.rescale(x)
        return x

    def forward(self, x):
        z = self.features(x)
        out = self.g(z)
        return out


def setup_optimizerNscheduler(args, model, distil=False):
        if distil: 
              optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.optim_wd, 
                                momentum=args.optim_mom,
                                nesterov=args.optim_nesterov)
              scheduler = None
              return optimizer, scheduler
        
        lr = args.lr/10
        if not args.optim_adam: # standard optimizer config
                optimizer = torch.optim.SGD(model.parameters(), 
                                lr=lr, 
                                weight_decay=args.optim_wd, 
                                momentum=args.optim_mom,
                                nesterov=args.optim_nesterov)
        else: 
                optimizer = torch.optim.Adam(model.parameters(), 
                                             lr = lr, 
                                             weight_decay=args.optim_wd)
                
        if not args.optim_cosineanneal: 
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        else: 
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs-args.optim_warmup)

        if args.optim_warmup > 0: # initialise warmup scheduler
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                                     start_factor=0.01, 
                                                                     total_iters=args.optim_warmup)
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
                                                                schedulers=[warmup_scheduler, scheduler], 
                                                                milestones=[args.optim_warmup])
        return optimizer, scheduler
        

def load_checkpoint(best=False, filename='checkpoint.pth.tar', distributed=False):
    path = base_path() + "chkpts" + "/" + "cifar10" + "/" + "rn50-ssl/"
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
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--checkpoints', action='store_true', help='Storing a checkpoint at every epoch. Loads a checkpoint if present.')
    parser.add_argument('--pretrained', action='store_true', help='Using a pre-trained network instead of training one.')
    parser.add_argument('--optim_wd', type=float, default=1e-3, help='optimizer weight decay.')
    parser.add_argument('--optim_adam', default=False, action='store_true', help='Using the Adam optimizer instead of SGD.')
    parser.add_argument('--optim_mom', type=float, default=0, help='optimizer momentum.')
    parser.add_argument('--optim_warmup', type=int, default=5, help='Number of warmup epochs.')
    parser.add_argument('--optim_nesterov', default=False, action='store_true', help='optimizer nesterov momentum.')
    parser.add_argument('--optim_cosineanneal', default=True, action='store_true', help='Enabling cosine annealing of learning rate..')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs.')
    parser.add_argument('--n_epochs_distil', type=int, default=60, help='Number of student epochs.')
    parser.add_argument('--batch_size', type=int, default = 256, help='Batch size.')
    parser.add_argument('--validate_subset', type=int, default=-1, 
                        help='If positive, allows validating on random subsets of the validation dataset during training.')

    add_management_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--alpha', type=float, default=0.5, required=True,
                        help='The weight of labels vs logits in the distillation loss (when alpha=1 only true labels are used)')
    parser.add_argument('--MSE', default=False, action='store_true',
                        help='If provided, the MSE loss is used for the student with labels .')
    parser.add_argument('--distillation_type', type=str, default='vanilla', choices=['vanilla', 'topK', 'inner', 'inner-parallel', 'topbottomK','randomK'],
                        help='Selects the distillation type, which determines the distillation loss.')
    parser.add_argument('--K', type=int, default=100, help='Number of activations to look at for *topK* distillation loss.')
    parser.add_argument('--N_BLOCKS', type=int, default=1, help='Number of layer blocks to distill from. The layers are selected in a reverse ordering from the output to input.')
    parser.add_argument('--gamma', type=float, default=1.0, help='The mixing weight for mixed inner distillation')
    args = parser.parse_args()

    return args


args = parse_args()
# Add uuid, timestamp and hostname for logging
args.conf_jobnum = str(uuid.uuid4())
args.conf_timestamp = str(datetime.datetime.now())
args.conf_host = socket.gethostname()

if args.seed is not None:
        set_random_seed(args.seed)

# dataset ->cifar5m
print(f"Randomly drawing {TEACHER_DATA_SIZE} samples for the Cifar5M base")
C5m_train, C5m_test = load_dataset('cifar5m', augment=AUGMENT)
all_indices = set(range(len(C5m_train)))
random_indices = np.random.choice(list(all_indices), 
                size=TEACHER_DATA_SIZE, replace=False)
teacher_train_subset = Subset(C5m_train, random_indices)

# initialising the model
teacher = TeacherNet(feature_dim=128) 

setproctitle.setproctitle('{}_{}_{}'.format("rn50ssl", args.buffer_size if 'buffer_size' in args else 0, "cifar5m"))

# start the training 
print(args)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(d) for d in args.gpus_id])
if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if args.wandb_name is None: 
                name = str.join("-",["offline", "cifar5m", "mnetSSL", args.conf_timestamp])
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
train_loader = DataLoader(teacher_train_subset, batch_size=args.batch_size, 
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(C5m_test, batch_size=args.batch_size, 
                        shuffle=False, num_workers=4, pin_memory=False)


checkpoint = load_checkpoint(best=False, filename=CHKPT_NAME, distributed=not args.distributed=='no') 
teacher.load_state_dict(checkpoint['state_dict'])
teacher.to(device)
start_epoch = checkpoint['epoch']
final_val_acc_D = checkpoint['best_acc']
        
df = {'final_val_acc_D':final_val_acc_D}
wandb.log(df)


print(f"Randomly drawing {args.buffer_size} samples from Cifar5M")
teacher.eval() # set the main model to evaluation
all_indices = set(range(len(C5m_train)))
random_indices = np.random.choice(list(all_indices), 
                size=args.buffer_size, replace=False)

train_subset = Subset(C5m_train, random_indices)
buffer_loader =  DataLoader(train_subset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=2,  
                            pin_memory=False)

args = parse_args(buffer=True)
experiment_log = vars(args)
experiment_log['final_val_acc_D'] = final_val_acc_D


student = StudentNet()
student.to(device)
student.train()
model_parameters = filter(lambda p: p.requires_grad, student.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"Student network instantiated with {params} parameters")



if args.alpha == 0: 
        print("Starting student SSL training ... ")
        start = time.time()
        optimizer, scheduler = setup_optimizerNscheduler(args, student, distil=True)
        for e in range(args.n_epochs_distil):
                if args.debug_mode and e > 3: # only 3 batches in debug mode
                        break
                avg_loss, total = 0.0, 0
                for i, data in enumerate(buffer_loader):
                        if args.debug_mode and i>10:
                                break
                        inputs, _ = data
                        inputs = inputs.to(device)
                        
                        with torch.no_grad(): f_t, o_t = teacher(inputs)
                        optimizer.zero_grad()
                        f_s = student.features(inputs)
                        total += inputs.size(0)
                        # the distillation loss
                        loss = vanilla_distillation(f_t, f_s)
                        loss.backward()
                        optimizer.step()

                        assert not math.isnan(loss)
                        progress_bar.prog(i, len(buffer_loader), e, 'SSL training', loss.item())
                        avg_loss += loss*inputs.size(0)
                
                avg_loss = avg_loss.item()/total
                if scheduler is not None:
                        scheduler.step()

                print('\nDistillation Loss : {} %'.format(round(avg_loss, 2)), file=sys.stderr)
                
        
        end = time.time()
        print(f"SSL training completed in {end-start} s.")

        student.freeze_backbone()
        experiment_log['SSL_training_loss'] = avg_loss

print("Starting CIFAR 5m training ... ")
optimizer, scheduler = setup_optimizerNscheduler(args, student, distil=False)

start = time.time()
alpha = args.alpha
for e in range(args.n_epochs):
        if args.debug_mode and e > 3: # only 3 batches in debug mode
                break
        avg_loss = 0.0
        best_acc = 0.0
        correct, total = 0.0, 0.0
        for i, data in enumerate(buffer_loader):
                if args.debug_mode and i>10:
                       break
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = student(inputs)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                # the labels loss 
                if args.MSE: 
                      labels_loss = F.mse_loss(outputs, F.one_hot(labels, num_classes=10).to(torch.float))  
                else:
                      labels_loss = F.cross_entropy(outputs, labels)
                loss = labels_loss 
                loss.backward()
                optimizer.step()

                assert not math.isnan(loss)
                progress_bar.prog(i, len(buffer_loader), e, 'Student', loss.item())
                avg_loss += loss*inputs.size(0)
        
        avg_loss = avg_loss/total
        if scheduler is not None:
                scheduler.step()
        
        train_acc = (correct/total) * 100
        # measure distance in parameter space between the teacher and student models 
        val_acc = evaluate(student, val_loader, device, num_samples=args.validate_subset)
        is_best = val_acc > best_acc 


        print('\nTrain accuracy : {} %'.format(round(train_acc, 2)), file=sys.stderr)
        print('\Val accuracy : {} %'.format(round(val_acc, 2)), file=sys.stderr)
        
        df = {'epoch_loss_S':avg_loss,
              'epoch_train_acc_S':train_acc,
              'epoch_val_acc_S':val_acc}
        wandb.log(df)


end = time.time()
print(f"Training completed in {end -start} s. Full evaluation and logging...")

experiment_log['buffer_train_time'] = end-start
experiment_log['final_train_acc_S'] = train_acc

val_acc = evaluate(student, val_loader, device)
 
experiment_log['final_val_acc_S'] = val_acc


if not args.nowand:
        wandb.finish()


# dumping everything into a log file
path = base_path() + "results" + "/" + "cifar5m" + "/" + "mnetSSL" 
if not os.path.exists(path): os.makedirs(path)
with open(path+ "/logs.txt", 'a') as f:
        f.write(json.dumps(experiment_log) + '\n')