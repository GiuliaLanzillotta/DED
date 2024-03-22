# Giulia Lanzillotta . 04.07.2023
# We train using distillation on a different dataset than what the teacher had been trained on. 

#Resources
# Oxford-Pet dataset https://www.robots.ox.ac.uk/~vgg/data/pets/ 
# https://github.com/limalkasadith/OxfordIIITPet-classification/blob/main/Linear%20Classification.ipynb

# Food-101 dataset https://pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html#torchvision.datasets.Food101 
# Hyperparameters https://github.com/hwchen2017/resnet_food101_cifar10_pytorch/blob/main/food101_resnet.py 


"""
example commands: 

python scripts/transfer.py --pretrained --alpha 0 --gpus_id 2 --buffer_size 3680  --notes transfer-pet --wandb_project DataEfficientDistillation
python scripts/transfer.py --data food --n_epochs 40 --n_epochs_stud 40 --lr 0.1  --pretrained --alpha 1 --gpus_id 5 --buffer_size 75750  --notes transfer-food --wandb_project DataEfficientDistillation

"""

from copy import deepcopy
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
from torchvision.datasets import ImageNet, ImageFolder
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18
import torchvision.transforms as transforms
from torch.nn.utils import parameters_to_vector


import shutil
from utils.args import add_management_args, add_rehearsal_args
from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar
from utils.stil_losses import *
from utils.nets import *
from utils.eval import evaluate, validation_and_agreement, distance_models, validation_agreement_function_distance, evaluate_CKA_teacher, evaluate_FA_teacher, evaluate_CKAandFA_teacher
from dataset_utils.data_utils import load_dataset

try:
    import wandb
except ImportError:
    wandb = None


AUGMENT = True
LOGITS_MAGNITUDE_TEACHER = 1.
CHKPT_NAME = "rn50_2023-02-21_10-45-30_best.ckpt"


def setup_optimizerNscheduler(args, model, data="pet"):

        if data == "pet": 
              optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.optim_wd, 
                                momentum=args.optim_mom)
              scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=4, mode='min', verbose=True)

        elif data == "food": 
                optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.optim_wd, 
                                momentum=args.optim_mom)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 16, 24], gamma=0.2)

        return optimizer, scheduler
                        

              

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', net="resnet50", data="pet"):
    path = base_path() + "/chkpts" + "/" + "transfer" + "/" + f"{data}/" + f"{net}/"
    if not os.path.exists(path): os.makedirs(path)
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(filename, path+'teacher_best.pth.tar')

def load_teacher_checkpoint(best=False, filename='checkpoint.pth.tar', distributed=False):
    path = base_path() + "chkpts" + "/" + "imagenet" + "/" + "resnet50/"
    if best: filepath = path + 'teacher_best.pth.tar'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath)
          if filename==CHKPT_NAME and not distributed: # modify Sidak's checkpoint
                new_state_dict = {k.replace('module.','',1):v for (k,v) in checkpoint['state_dict'].items()}
                checkpoint['state_dict'] = new_state_dict
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
    parser.add_argument('--optim_wd', type=float, default=1e-4, help='optimizer weight decay.')
    parser.add_argument('--optim_adam', default=False, action='store_true', help='Using the Adam optimizer instead of SGD.')
    parser.add_argument('--optim_mom', type=float, default=0.9, help='optimizer momentum.')
    parser.add_argument('--optim_warmup', type=int, default=0, help='Number of warmup epochs.')
    parser.add_argument('--optim_nesterov', type=int, default=0, help='optimizer nesterov momentum.')
    parser.add_argument('--optim_cosineanneal', default=False, action='store_true', help='Enabling cosine annealing of learning rate..')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--n_epochs_stud', type=int, default=100, help='Number of student epochs.')
    parser.add_argument('--batch_size', type=int, default = 64, help='Batch size.')
    parser.add_argument('--validate_subset', type=int, default=-1, 
                        help='If positive, allows validating on random subsets of the validation dataset during training.')
    
    add_management_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--temperature', type=float, default=1., help='Temperature (prop to entropy) of the teacher outputs - only used with KL.')
    parser.add_argument('--alpha', type=float, default=0.5, required=True,
                        help='The weight of labels vs logits in the distillation loss (when alpha=1 only true labels are used)')
    parser.add_argument('--MSE', default=False, action='store_true',
                        help='If provided, the MSE loss is used for the student with labels .')
    parser.add_argument('--data',type=str, default="pet",help='Dataset for transfer.')
    parser.add_argument('--lowdatateacher', default=False, action='store_true',
                        help='If provided, the teacher is also fine-tuned on a low amount of data.')

    args = parser.parse_args()
    return args



args = parse_args()
# Add uuid, timestamp and hostname for logging
args.conf_jobnum = str(uuid.uuid4())
args.conf_timestamp = str(datetime.datetime.now())
args.conf_host = socket.gethostname()

if args.seed is not None:
        set_random_seed(args.seed)



setproctitle.setproctitle('{}_{}_{}'.format("resnet50", args.buffer_size if 'buffer_size' in args else 0, "transfer"))

# start the training 
print(args)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(d) for d in args.gpus_id])
if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if args.wandb_name is None: 
                name = str.join("-",["offline", "transfer", "resnet50", args.conf_timestamp])
        else: name = args.wandb_name
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
                        name=name, notes=args.notes, config=vars(args)) 
        args.wandb_url = wandb.run.get_url()


device = get_device([0]) # returns the first device in the list

train_dataset, val_dataset = load_dataset(args.data, augment=AUGMENT)
if args.data=='pet': num_classes = 37
if args.data=='food': num_classes = 101

# initialising the teacher
# teacher_base = resnet50(weights=None)
# checkpoint = load_teacher_checkpoint(best=False, filename=CHKPT_NAME, distributed=not args.distributed=='no') 
# teacher_base.load_state_dict(checkpoint['state_dict'])
# teacher_base.eval()
# we here add a linear classifier on top of the teacher
# classifier = nn.Linear(1000, num_classes)
# teacher = nn.Sequential(teacher_base, classifier)
# teacher.to(device)
# print(teacher)

# for i, m in enumerate(teacher.children()):
#         if i ==0:
#                 print("Freezing the first module ([0])")
#                 for p in m.parameters():
#                         p.requires_grad = False
#         else: print(f"Not freezing {i}")

teacher = resnet50(weights=None)
checkpoint = load_teacher_checkpoint(best=False, filename=CHKPT_NAME, distributed=not args.distributed=='no') 
teacher.load_state_dict(checkpoint['state_dict'])
teacher.eval()

for (n,m) in teacher.named_children():
       if n != "fc":
              print("Freezing ",n)
              for p in m.parameters():
                     p.requires_grad = False

print("Restoring classifier")
teacher.fc = nn.Linear(2048, num_classes)

teacher.to(device)
params = count_parameters(teacher)
print(f"Linear classifier created with {params} parameters")


progress_bar = ProgressBar(verbose=not args.non_verbose)

print(file=sys.stderr)
train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)


# Moving on to training the student 
print(f"Randomly drawing {args.buffer_size} samples from {args.data} dataset")
all_indices = set(range(len(train_dataset)))
random_indices = np.random.choice(list(all_indices), size=args.buffer_size, replace=False)
#left_out_indices = all_indices.difference(set(random_indices.flatten()))
train_subset = Subset(train_dataset, random_indices)
buffer_loader =  DataLoader(train_subset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=True)
# train_leftout_subset = Subset(train_dataset, list(left_out_indices))
# train_leftout_loader = DataLoader(train_leftout_subset, 
#                                 batch_size=args.batch_size, 
#                                 shuffle=False, 
#                                 num_workers=4, 
#                                 pin_memory=False)

if args.lowdatateacher:
        train_loader = buffer_loader

#training the teacher head on the new dataset here 
optimizer, scheduler = setup_optimizerNscheduler(args, teacher, data=args.data)
val_acc=0.0

if args.pretrained: 
        print("Attempting to load teacher+classifier from file")
        path = base_path() + "chkpts" + "/" + "transfer" + "/" + f"{args.data}" + "/" + "resnet50/"
        filepath = path + f"teacher.ckpt"
        if args.lowdatateacher:
                filepath = path + f"teacher_{args.buffer_size}.ckpt"
        if os.path.exists(filepath):
                print(f"Loading existing checkpoint {filepath}")
                checkpoint = torch.load(filepath)
                teacher.load_state_dict(checkpoint['state_dict'])

if (not args.pretrained) or (not os.path.exists(filepath)):
        print(f"Finetuning teacher+classifier to {args.data} dataset")
        for e in range(args.n_epochs):
                if args.debug_mode and e > 3: # only 3 batches in debug mode
                        break
                avg_loss = 0.0
                correct, total = 0.0, 0.0
                for i, data in enumerate(train_loader):
                        if args.debug_mode and i>10:
                                break
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        
                        optimizer.zero_grad()
                        outputs = teacher(inputs)
                        
                        _, pred = torch.max(outputs.data, 1)
                        correct += torch.sum(pred == labels).item()
                        total += labels.shape[0]
                        
                        loss = F.cross_entropy(outputs, labels, reduction='mean')

                        loss.backward()
                        optimizer.step()
                        assert not math.isnan(loss)
                        progress_bar.prog(i, len(train_loader), e, val_acc, loss.item())
                        avg_loss += loss
                
                avg_loss = avg_loss/i
                correct = correct/total

                val_acc = evaluate(teacher, val_loader, device)

                if scheduler is not None:
                        scheduler.step(val_acc)

                
                print('\nTrain accuracy : {} %'.format(round(correct*100, 2)), file=sys.stderr)
                print('\Val accuracy : {} %'.format(round(val_acc, 2)), file=sys.stderr)
        

        print("Classifier training finished.")
        filename = "teacher.ckpt"
        if args.lowdatateacher:
                filename = f"teacher_{args.buffer_size}.ckpt"
        save_checkpoint({
        'epoch': e + 1,
        'state_dict': teacher.state_dict(),
        'best_acc': val_acc,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
        }, 
        False, 
        net="resnet50",
        data=args.data,
        filename=filename)



teacher.eval() # set the teacher to evaluation

val_acc = evaluate(teacher, val_loader, device)
df = {'final_val_acc_D':val_acc}
wandb.log(df)
experiment_log = vars(args)
experiment_log['final_val_acc_D'] = val_acc




print("Starting student training ... ")
start = time.time()

net = "resnet18"

if net=="resnet18":
        student = resnet18(num_classes)
student = feature_wrapper(student) # add 'get_features' function
student = head_wrapper(student) # add 'forward_head' function

params = count_parameters(student)
print(f"Student created with {params} parameters")

student.to(device)
student.train()

student_init = deepcopy(student)
student_init.eval()

optimizer, scheduler = setup_optimizerNscheduler(args, student, data=args.data)


alpha = args.alpha
T = args.temperature
val_acc = 0.0
for e in range(args.n_epochs_stud):
        if args.debug_mode and e > 3: # only 3 batches in debug mode
                break
        avg_loss = 0.0
        best_acc = 0.0
        correct, total, agreement, function_distance = 0.0, 0.0, 0.0, 0.0
        for i, data in enumerate(buffer_loader):
                if args.debug_mode and i>10:
                       break
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.no_grad(): 
                       logits = teacher(inputs)

                optimizer.zero_grad()
                outputs = student(inputs)
                
                _, pred = torch.max(outputs.data, 1)
                _, pred_t = torch.max(logits.data, 1)
                correct += torch.sum(pred == labels).item()
                agreement += torch.sum(pred == pred_t).item()
                total += labels.shape[0]

                if args.MSE: 
                        labels_loss = F.mse_loss(outputs, F.one_hot(labels, num_classes=num_classes).to(torch.float) * LOGITS_MAGNITUDE_TEACHER, reduction='none').mean(dim=1)  # Bobby's correction
                        logits_loss = F.mse_loss(outputs, logits, reduction='none').mean(dim=1)
                else:
                        labels_loss = F.cross_entropy(outputs, labels, reduction='none')
                        logits_loss = F.kl_div(input=F.log_softmax(outputs/T, dim=1), target=F.softmax(logits/T, dim=1), log_target=False, reduction='none').sum(dim=1) * (T**2) # temperature rescaling (for gradients)
                

                loss = alpha*labels_loss.mean() + (1-alpha)*logits_loss.mean()
                
                loss.backward()
                optimizer.step()
                assert not math.isnan(loss)
                progress_bar.prog(i, len(buffer_loader), e, 'S', loss.item())
                avg_loss += loss
        
        avg_loss = avg_loss/i
        if scheduler is not None:
                if args.data=="pet": scheduler.step(val_acc)
                else: scheduler.step()
        
        train_acc = (correct/total) * 100
        train_agreement = (agreement/total) * 100

        # measure distance in parameter space between the teacher and student teachers 
        val_acc, val_agreement = validation_and_agreement(student, teacher, val_loader, device, num_samples=args.validate_subset)

        fa, cka = evaluate_CKAandFA_teacher(teacher, student, buffer_loader, device, batches=10)
        fa_init, cka_init = evaluate_CKAandFA_teacher(student_init, student, buffer_loader, device, batches=10)

        print('\nTrain accuracy : {} %'.format(round(train_acc, 2)), file=sys.stderr)
        print('\Val accuracy : {} %'.format(round(val_acc, 2)), file=sys.stderr)
        
        df = {'epoch_loss_S':avg_loss,
                'epoch_train_acc_S':train_acc,
                'epoch_train_agreement':train_agreement,
                'epoch_val_acc_S':val_acc,
                'epoch_val_agreement':val_agreement,
                'epoch_cka_train':cka,
                'epoch_fa_train':fa,
                'epoch_cka_train_init':cka_init,
                'epoch_fa_train_init':fa_init,
                'lr':scheduler._last_lr
                }
        wandb.log(df)


print("Training completed. Full evaluation and logging...")
end = time.time()

if args.checkpoints_stud: 
        save_checkpoint({
        'epoch': e + 1,
        'state_dict': student.state_dict(),
        'best_acc': val_acc,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
        }, 
        False, 
        filename=f'{net}-student-{args.seed}-{args.buffer_size}-{args.alpha}-{args.temperature}.ckpt',
        net=net, 
        data=args.data)

fa_train, cka_train = evaluate_CKAandFA_teacher(teacher, student, buffer_loader, device, batches=20)
fa_val, cka_val = evaluate_CKAandFA_teacher(teacher, student, val_loader, device, batches=10)


fa_train_init, cka_train_init = evaluate_CKAandFA_teacher(student_init, student, buffer_loader, device, batches=20)
fa_val_init, cka_val_init = evaluate_CKAandFA_teacher(student_init, student, val_loader, device, batches=10)


experiment_log['final_cka_train'] = cka_train
experiment_log['final_cka_val'] = cka_val
experiment_log['final_fa_train'] = fa_train
experiment_log['final_fa_val'] = fa_val
experiment_log['final_fa_train_init'] = fa_train_init
experiment_log['final_fa_val_init'] = fa_val_init
experiment_log['buffer_train_time'] = end-start
experiment_log['final_train_acc_S'] = train_acc

val_acc, val_agreement = validation_and_agreement(student, teacher, val_loader, device)

experiment_log['final_val_acc_S'] = val_acc
experiment_log['final_train_agreement'] = train_agreement
experiment_log['final_val_agreement'] = val_agreement

experiment_log['net'] = net



if not args.nowand:
        wandb.finish()


# dumping everything into a log file
path = base_path() + "results" + "/" + "transfer" + "/" + f"{args.data}" + "/" + f"{net}"
if not os.path.exists(path): os.makedirs(path)
with open(path+ "/logs.txt", 'a') as f:
        f.write(json.dumps(experiment_log) + '\n')
