# Giulia Lanzillotta . 17.10.2023
# Cifar 5M training experiment script 

"""

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
We use cifar5m, an extension to 5 mln images in order to train the student on more images than the teacher. 


example commands: 

python scripts/cifar5m_2heads.py --MSE --seed 11 --beta 0.5  --gpus_id 7 --buffer_size 60000 --distillation_type vanilla --batch_size 128  --checkpoints --notes cifar5m-distillation-2heads --wandb_project DataEfficientDistillation



"""

import copy
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
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18, googlenet, efficientnet_b0, mobilenet_v3_large
import torchvision.transforms as transforms
from torch.nn.utils import parameters_to_vector


import shutil
from utils.args import add_management_args, add_rehearsal_args
from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar
from utils.stil_losses import *
from utils.nets import *
from utils.eval import evaluate, validation_and_agreement, distance_models, validation_agreement_function_distance
from dataset_utils.data_utils import load_dataset

try:
    import wandb
except ImportError:
    wandb = None

LOGITS_MAGNITUDE_TEACHER = 1.0 #TODO
AUGMENT = True


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class TwoHeadsNet(nn.Module):
    # a mobilenet with two heads
    def __init__(self, num_classes=10, beta=1.0, c=20):
        """ Beta is the mixing factor for the models predictions."""
        super(TwoHeadsNet, self).__init__()

        layer1 = nn.Sequential(
                nn.Conv2d(3, c, kernel_size=3, stride=1,padding=1, bias=True),
                nn.ReLU())
        layer2 = nn.Sequential(nn.Conv2d(c, c*2, kernel_size=3, stride=1, padding=1, bias=True),
                               nn.BatchNorm2d(c*2),
                               nn.ReLU(),
                               nn.MaxPool2d(2))
        layer3 = nn.Sequential(nn.Conv2d(c*2, c*4, kernel_size=3,
                  stride=1, padding=1, bias=True),
                nn.BatchNorm2d(c*4),
                nn.ReLU(),
                nn.MaxPool2d(2))
        layer4 = nn.Sequential(nn.Conv2d(c*4, c*8, kernel_size=3,
                  stride=1, padding=1, bias=True),
                nn.BatchNorm2d(c*8),
                nn.ReLU(),
                nn.MaxPool2d(2))
        final_layer = nn.Sequential(nn.MaxPool2d(4),
                        Flatten(),
                        nn.Linear(c*8, num_classes, bias=True))

        # we divide the backbone from the heads ----------------------
        self.f = nn.Sequential(layer1, layer2, layer3) 

        self.labels_head = nn.Sequential(
         copy.deepcopy(layer4), copy.deepcopy(final_layer)
        )
        self.logits_head = nn.Sequential(
         copy.deepcopy(layer4), copy.deepcopy(final_layer)
        )

        self.beta = beta

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"{'Student'} CNN made with {params} parameters")

        b_params = sum([np.prod(p.size()) for p in self.f.parameters()])
        print(f"{'Student'} : {b_params} parameters in the backbone")


    def forward(self, x, mixed=True):
        z = self.f(x)
        labels_out = self.labels_head(z)
        logits_out = self.logits_head(z)

        if mixed: return self.beta*labels_out + (1-self.beta)*logits_out

        return labels_out, logits_out


def validation_2heads(student:TwoHeadsNet, val_loader, device, num_samples=-1):
        """ Like evaluate, but it also returns the average agreement of student and teacher"""
        status = student.training
        student.eval()
        teacher.eval() # shouldn't be needed
        if num_samples >0: 
                # we select a subset of the validation dataset to validate on 
                # note: a different random sample is used every time
                random_indices = np.random.choice(range(len(val_loader.dataset)), size=num_samples, replace=False)
                _subset = Subset(val_loader.dataset, random_indices)
                val_loader =  DataLoader(_subset, 
                                        batch_size=val_loader.batch_size, 
                                        shuffle=False, num_workers=4, pin_memory=False)
        correct_l, correct_d, total = 0.0, 0.0, 0.0
        for i,data in enumerate(val_loader):
                with torch.no_grad():
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs_s_l, outputs_s_d = student.forward(inputs, mixed=False)
                        _, pred_s_l = torch.max(outputs_s_l.data, 1)
                        _, pred_s_d = torch.max(outputs_s_d.data, 1)
                        correct_l += torch.sum(pred_s_l == labels).item()
                        correct_d += torch.sum(pred_s_d == labels).item()
                        total += labels.shape[0]
                                
        acc_l=(correct_l / total) * 100
        acc_d=(correct_d / total) * 100
        student.train(status)
        return acc_l, acc_d

def setup_optimizerNscheduler(args, model, stud=False):
        if stud: epochs = args.n_epochs_stud
        else: epochs = args.n_epochs
        if not args.optim_adam:
                optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.optim_wd, 
                                momentum=args.optim_mom,
                                nesterov=args.optim_nesterov)
        else: 
                optimizer = torch.optim.Adam(model.parameters(), 
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
    path = base_path() + "/chkpts" + "/" + "cifar5m" + "/" + "convnet/"
    if not os.path.exists(path): os.makedirs(path)
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(path+filename, path+'model_best.ckpt')

def load_checkpoint(best=False, filename='checkpoint.pth.tar', distributed=False):
    path = base_path() + "chkpts" + "/" + "cifar5m" + "/" + "convnet/"
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
    parser.add_argument('--n_epochs_stud', type=int, default=30, help='Number of student epochs.')
    parser.add_argument('--batch_size', type=int, default = 256, help='Batch size.')
    parser.add_argument('--validate_subset', type=int, default=-1, 
                        help='If positive, allows validating on random subsets of the validation dataset during training.')

    add_management_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--beta', type=float, default=0.5, required=True,
                        help='The weight of labels vs logits in the prediction (when alpha=1 only true labels are used)')
    
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

# dataset -> cifar100 for the teacher and cifar5m for the student
#C10_train, C10_val = load_dataset('cifar10', augment=AUGMENT)

C5m_train, C5m_test = load_dataset('cifar5m', augment=AUGMENT)


print(f"Randomly drawing {60000} samples for the Cifar5M base")
all_indices = set(range(len(C5m_train)))
random_indices = np.random.choice(list(all_indices), size=60000, replace=False)
teacher_data = Subset(C5m_train, random_indices)

# initialising the model
teacher = make_cnn(c=20, num_classes=10, use_batch_norm=True) # adjusting for CIFAR 

setproctitle.setproctitle('{}_{}_{}'.format("convnet-2heads", args.buffer_size if 'buffer_size' in args else 0, "imagenet"))

# start the training 
print(args)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(d) for d in args.gpus_id])
if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if args.wandb_name is None: 
                name = str.join("-",["cifar5m", "convnet-2heads", args.conf_timestamp])
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
train_loader = DataLoader(teacher_data, batch_size=args.batch_size, 
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(C5m_test, batch_size=args.batch_size, 
                        shuffle=False, num_workers=4, pin_memory=False)



CHKPT_NAME = f'convnet-teacher.ckpt' # obtaineed with seed = 11

if not args.pretrained:
        teacher.train()
        optimizer, scheduler = setup_optimizerNscheduler(args, teacher)
        best_acc = 0.
        start_epoch = 0
        epoch=0
        is_best = True


        if args.checkpoints: # resuming training from the last point
                checkpoint = load_checkpoint(best=False, filename=CHKPT_NAME, 
                                             distributed=not args.distributed=='no') 
                if checkpoint: 
                        teacher.load_state_dict(checkpoint['state_dict'])
                        teacher.to(device)
                        #optimizer.load_state_dict(checkpoint['optimizer'])
                        scheduler.load_state_dict(checkpoint['scheduler'])
                        start_epoch = checkpoint['epoch']
                        val_acc = checkpoint['best_acc']
                        best_acc = val_acc


        for epoch in range(start_epoch, args.n_epochs):
                avg_loss = 0.0
                correct, total = 0.0, 0.0
                if args.debug_mode and epoch > 3:
                       break
                for i, data in enumerate(train_loader):
                        optimizer.zero_grad()
                        if args.debug_mode and i > 3: # only 3 batches in debug mode
                                break
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = teacher(inputs)
                        with torch.no_grad():
                                _, pred = torch.max(outputs, 1)
                                correct += torch.sum(pred == labels).item()
                        loss = F.cross_entropy(outputs, labels) #TODO: maybe MSE?
                        loss.backward()
                        optimizer.step()

                        assert not math.isnan(loss)
                        progress_bar.prog(i, len(train_loader), epoch, 'Teacher', loss.item())
                        avg_loss += loss
                        total += labels.shape[0]

                if scheduler is not None:
                        scheduler.step()
                
                train_acc = correct/total * 100
                val_acc = evaluate(teacher, val_loader, device, num_samples=args.validate_subset)

                # best val accuracy -> selection bias on the validation set
                is_best = val_acc > best_acc
                best_acc = max(val_acc, best_acc)

                print('\Train accuracy : {} %'.format(round(train_acc, 2)), file=sys.stderr)
                print('\Val accuracy : {} %'.format(round(val_acc, 2)), file=sys.stderr)
                
                df = {'epoch_loss_D':avg_loss/total,
                'epoch_train_acc_D':train_acc,
                'epoch_val_acc_D':val_acc}
                wandb.log(df)


        if args.checkpoints and start_epoch<args.n_epochs: 
                save_checkpoint({
                'epoch': args.n_epochs + 1,
                'state_dict': teacher.state_dict(),
                'best_acc': val_acc,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
                }, is_best, filename=CHKPT_NAME)
                
        final_val_acc_D = val_acc
else: 
        checkpoint = load_checkpoint(best=False, filename=CHKPT_NAME, distributed=not args.distributed=='no') #TODO: switch best off
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.to(device)
        final_val_acc_D = checkpoint['best_acc']

df = {'final_val_acc_D':final_val_acc_D}
wandb.log(df)
teacher.eval() # set the main model to evaluation


print(f"Randomly drawing {args.buffer_size} samples for the Cifar5M base")

random_indices = np.random.choice(list(all_indices), 
                size=args.buffer_size, replace=False)

student_data = Subset(C5m_train, random_indices)
buffer_loader =  DataLoader(student_data, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=2,  
                            pin_memory=False)

args = parse_args(buffer=True)
experiment_log = vars(args)
experiment_log['backbone_layers'] = 3
experiment_log['final_val_acc_D'] = final_val_acc_D

wandb.config['backbone_layers']=3


print("Starting student training ... ")
start = time.time()
# re-initialise model 
student = TwoHeadsNet(num_classes=10, beta=args.beta)
student.to(device)
student.train()

optimizer, scheduler = setup_optimizerNscheduler(args, student, stud=True)


average_magnitude = 0
beta=args.beta
for e in range(args.n_epochs):
        if args.debug_mode and e > 3: # only 3 batches in debug mode
                break
        avg_loss, avg_l_loss, avg_d_loss = 0.0, 0.0, 0.0
        best_acc = 0.0
        correct_labels, correct_distil, total, agreement = 0.0, 0.0, 0.0, 0.0
        for i, data in enumerate(buffer_loader):
                if args.debug_mode and i>10:
                       break
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.no_grad(): logits = teacher(inputs)
                optimizer.zero_grad()
                out_labels, out_distil = student.forward(inputs, mixed=False)
                _, pred_labels = torch.max(out_labels.data, 1)
                _, pred_distil = torch.max(out_distil.data, 1)
                _, pred_t = torch.max(logits.data, 1)
                correct_labels += torch.sum(pred_labels == labels).item()
                correct_distil += torch.sum(pred_distil == labels).item()
                agreement += torch.sum(pred_distil == pred_t).item()
                total += labels.shape[0]
                
                # the distillation loss
                logits_loss = vanilla_distillation(out_distil, logits)
                # the labels loss 
                if args.MSE: 
                      labels_loss = F.mse_loss(out_labels, F.one_hot(labels, num_classes=10).to(torch.float) * LOGITS_MAGNITUDE_TEACHER)  # Bobby's correction
                else:
                      labels_loss = F.cross_entropy(out_labels, labels)
                loss = beta*labels_loss + (1-beta)*logits_loss
                loss.backward()
                optimizer.step()        


                if LOGITS_MAGNITUDE_TEACHER == 1: # estimate during the first epoch
                       average_non_max = (logits.sum(dim=1) - logits.max(dim=1)[0])/9 # average over the non-max outputs
                       average_magnitude += (logits.max(dim=1)[0] - average_non_max).sum(dim=0) 
        

                assert not math.isnan(loss)
                progress_bar.prog(i, len(buffer_loader), e, 'Student', loss.item())
                avg_loss += loss*(labels.shape[0])
                avg_l_loss += labels_loss*(labels.shape[0])
                avg_d_loss += logits_loss+(labels.shape[0])

        
        avg_loss = avg_loss/total
        avg_l_loss = avg_l_loss/total
        avg_d_loss = avg_d_loss/total

        if scheduler is not None:
                scheduler.step()


        if LOGITS_MAGNITUDE_TEACHER==1: 
                average_magnitude = average_magnitude/total
                LOGITS_MAGNITUDE_TEACHER = average_magnitude
                print(f"Setting LMT to {LOGITS_MAGNITUDE_TEACHER}")
        

        train_acc_labels = (correct_labels/total) * 100
        train_acc_distil = (correct_distil/total) * 100
        train_agreement = (agreement/total) * 100      

        val_acc_labels, val_acc_distil = validation_2heads(student, val_loader, device, num_samples=args.validate_subset)

        is_best = val_acc_labels > best_acc 

        print('\nTrain accuracy (labels) : {} %'.format(round(train_acc_labels, 2)), file=sys.stderr)
        print('\nTrain accuracy (distil) : {} %'.format(round(train_acc_distil, 2)), file=sys.stderr)
        print('\Val accuracy (labels): {} %'.format(round(val_acc_labels, 2)), file=sys.stderr)
        print('\Val accuracy (distil): {} %'.format(round(val_acc_distil, 2)), file=sys.stderr)
        
        df = {'epoch_loss_S':avg_loss,
              'epoch_labels_loss_S':avg_l_loss,
              'epoch_distil_loss_S':avg_d_loss,
              'epoch_train_acc_labels_S':train_acc_labels,
              'epoch_train_acc_distil_S':train_acc_distil,
              'epoch_train_agreement':train_agreement,
              'epoch_val_acc_labels_S':val_acc_labels,
              'epoch_val_acc_distil_S':val_acc_distil}
        wandb.log(df)
        
end = time.time()

print(f"Training completed in {end-start} s. Full evaluation and logging...")
experiment_log['buffer_train_time'] = end-start
experiment_log['final_train_labels_S'] = train_acc_labels,
experiment_log['final_train_distil_S'] = train_acc_distil,
val_acc_labels, val_acc_distil = validation_2heads(student, val_loader, device)
val_acc_mixed = evaluate(student, val_loader, device)



experiment_log['final_val_acc_mixed'] = val_acc_mixed
experiment_log['final_val_acc_labels_S'] = val_acc_labels
experiment_log['final_val_acc_distil_S'] = val_acc_distil
experiment_log['final_train_agreement'] = train_agreement


if not args.nowand:
        wandb.finish()


# dumping everything into a log file
path = base_path() + "results" + "/" + "cifar5m" + "/" + "convnet-2heads" 
if not os.path.exists(path): os.makedirs(path)
with open(path+ "/logs.txt", 'a') as f:
        f.write(json.dumps(experiment_log) + '\n')