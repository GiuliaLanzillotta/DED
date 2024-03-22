# Giulia Lanzillotta . 04.07.2023
# Imagenet offline training experiment script (no continual structure)

#Resources
# https://github.com/pytorch/examples/blob/main/imagenet/main.py 
# https://pytorch.org/vision/0.8/datasets.html#imagenet
# https://www.image-net.org/about.php 

"""
example commands: 

python scripts/imagenet_vit.py --alpha 0 --gpus_id 0 --buffer_size 12000 --validate_subset 5000 --batch_size 256 --notes imagenet-vit --wandb_project DataEfficientDistillation




Teacher recipe: https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16
pytorch code repo https://github.com/jeonsworld/ViT-pytorch/blob/main/train.py

torchrun --nproc_per_node=8 train.py\
    --model vit_b_32 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment imagenet\
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema


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
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18, vit_b_16
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

AUGMENT = False
LOGITS_MAGNITUDE_TEACHER = 1


def setup_optimizerNscheduler(args, model, stud=False):

        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=0.003, 
                                      weight_decay=0.3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300-30)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.033, total_iters=30)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
                        schedulers=[warmup_scheduler, scheduler], milestones=[30])
        return optimizer, scheduler
                        

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    path = base_path() + "/chkpts" + "/" + "imagenet" + "/" + "vit/"
    if not os.path.exists(path): os.makedirs(path)
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(filename, path+'teacher_best.pth.tar')

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
    parser.add_argument('--n_epochs', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--n_epochs_stud', type=int, default=300, help='Number of student epochs.')
    parser.add_argument('--batch_size', type=int, default = 256, help='Batch size.')
    parser.add_argument('--validate_subset', type=int, default=-1, 
                        help='If positive, allows validating on random subsets of the validation dataset during training.')
    
    add_management_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--temperature', type=float, default=1., help='Temperature (prop to entropy) of the teacher outputs - only used with KL.')
    parser.add_argument('--alpha', type=float, default=0.5, required=True,
                        help='The weight of labels vs logits in the distillation loss (when alpha=1 only true labels are used)')
    parser.add_argument('--MSE', default=False, action='store_true',
                        help='If provided, the MSE loss is used for the student with labels .')
    parser.add_argument('--distillation_type', type=str, default='vanilla', choices=['vanilla', 'topK', 'inner', 'inner-parallel', 'topbottomK','randomK'],
                        help='Selects the distillation type, which determines the distillation loss.')
 
    args = parser.parse_args()
    return args


args = parse_args()
# Add uuid, timestamp and hostname for logging
args.conf_jobnum = str(uuid.uuid4())
args.conf_timestamp = str(datetime.datetime.now())
args.conf_host = socket.gethostname()

if args.seed is not None:
        set_random_seed(args.seed)


train_dataset, val_dataset = load_dataset('imagenet', augment=AUGMENT)

# initialising the teacher
teacher = vit_b_16(torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)

params = count_parameters(teacher)
print(f"Teacher created with {params} parameters")

setproctitle.setproctitle('{}_{}_{}'.format("vit", args.buffer_size if 'buffer_size' in args else 0, "imagenet"))

# start the training 
print(args)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(d) for d in args.gpus_id])
if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if args.wandb_name is None: 
                name = str.join("-",["offline", "imagenet", "vit", args.conf_timestamp])
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
train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

teacher.to(device)
final_val_acc_D = evaluate(teacher, val_loader, device, num_samples=args.validate_subset)
df = {'final_val_acc_D':final_val_acc_D}
wandb.log(df)

print(f"Randomly drawing {args.buffer_size} samples from ImageNet")
teacher.eval() # set the main teacher to evaluation
all_indices = set(range(len(train_dataset)))
random_indices = np.random.choice(list(all_indices), size=args.buffer_size, replace=False)
left_out_indices = all_indices.difference(set(random_indices.flatten()))
train_subset = Subset(train_dataset, random_indices)
buffer_loader =  DataLoader(train_subset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=True)
train_leftout_subset = Subset(train_dataset, list(left_out_indices))
train_leftout_loader = DataLoader(train_leftout_subset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=False)

experiment_log = vars(args)
experiment_log['final_val_acc_D'] = final_val_acc_D



print("Starting student training ... ")
start = time.time()
# re-initialise teacher 
student = vit_b_16(weights=None)
student = feature_wrapper(student) # add 'get_features' function
student = head_wrapper(student) # add 'forward_head' function

if args.distributed=='dp': 
      print(f"Parallelising buffer training on {len(args.gpus_id)} GPUs.")
      student = torch.nn.DataParallel(student, device_ids=args.gpus_id).to(device)
student.to(device)
student.train()

#student_init = deepcopy(student)
#student_init.eval()

optimizer, scheduler = setup_optimizerNscheduler(args, student, stud=True)

alpha = args.alpha
T = args.temperature
average_magnitude=0

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
                
                with torch.no_grad(): logits = teacher(inputs)
                optimizer.zero_grad()
                outputs = student(inputs)
                #if args.distillation_type=='inner-parallel': outputs.detach()
                
                _, pred = torch.max(outputs.data, 1)
                _, pred_t = torch.max(logits.data, 1)
                correct += torch.sum(pred == labels).item()
                agreement += torch.sum(pred == pred_t).item()
                total += labels.shape[0]
                

                if LOGITS_MAGNITUDE_TEACHER == 1 and args.MSE: # estimate during the first epoch
                       average_non_max = (logits.sum(dim=1) - logits.max(dim=1)[0])/9 # average over the non-max outputs
                       average_magnitude += (logits.max(dim=1)[0] - average_non_max).sum(dim=0) 
                

                # the labels loss 
                if args.MSE: 
                        labels_loss = F.mse_loss(outputs, F.one_hot(labels, num_classes=1000).to(torch.float) * LOGITS_MAGNITUDE_TEACHER, reduction='none').mean(dim=1)  # Bobby's correction
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
                scheduler.step()

        if LOGITS_MAGNITUDE_TEACHER==1 and args.MSE: 
               average_magnitude = average_magnitude/total
               LOGITS_MAGNITUDE_TEACHER = average_magnitude
               print(f"Setting LMT to {LOGITS_MAGNITUDE_TEACHER}")
        

        train_acc = (correct/total) * 100
        train_agreement = (agreement/total) * 100

        # measure distance in parameter space between the teacher and student teachers 
        teacher_student_distance = distance_models(teacher, student)
        val_acc, val_agreement = validation_and_agreement(student, teacher, val_loader, device, num_samples=args.validate_subset)

        fa, cka = evaluate_CKAandFA_teacher(teacher, student, buffer_loader, device, batches=10)
        #fa_init, cka_init = evaluate_CKAandFA_teacher(student_init, student, buffer_loader, device, batches=10)

        print('\nTrain accuracy : {} %'.format(round(train_acc, 2)), file=sys.stderr)
        print('\Val accuracy : {} %'.format(round(val_acc, 2)), file=sys.stderr)
        
        df = {'epoch_loss_S':avg_loss,
                'epoch_train_acc_S':train_acc,
                'epoch_train_agreement':train_agreement,
                'epoch_distance_teacher_student':teacher_student_distance,
                'epoch_val_acc_S':val_acc,
                'epoch_val_agreement':val_agreement,
                'epoch_cka_train':cka,
                'epoch_fa_train':fa,
                'epoch_cka_train_init':cka_init,
                'epoch_fa_train_init':fa_init,
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
        }, False, filename=f'rn50-student-{args.seed}-{args.buffer_size}-{args.alpha}-{args.temperature}.ckpt')

fa_train, cka_train = evaluate_CKAandFA_teacher(teacher, student, buffer_loader, device, batches=20)
fa_val, cka_val = evaluate_CKAandFA_teacher(teacher, student, val_loader, device, batches=10)


#fa_train_init, cka_train_init = evaluate_CKAandFA_teacher(student_init, student, buffer_loader, device, batches=20)
#fa_val_init, cka_val_init = evaluate_CKAandFA_teacher(student_init, student, val_loader, device, batches=10)


experiment_log['final_cka_train'] = cka_train
experiment_log['final_cka_val'] = cka_val
experiment_log['final_fa_train'] = fa_train
experiment_log['final_fa_val'] = fa_val
#experiment_log['final_fa_train_init'] = fa_train_init
#experiment_log['final_fa_val_init'] = fa_val_init
experiment_log['buffer_train_time'] = end-start
experiment_log['final_train_acc_S'] = train_acc

val_acc, val_agreement = validation_and_agreement(student, teacher, val_loader, device)

experiment_log['final_val_acc_S'] = val_acc
experiment_log['final_train_agreement'] = train_agreement
experiment_log['final_val_agreement'] = val_agreement
experiment_log['final_distance_teacher_student'] = teacher_student_distance

experiment_log['runs_id'] = "rebuttals"


if not args.nowand:
        wandb.finish()


# dumping everything into a log file
path = base_path() + "results" + "/" + "imagenet" + "/" + "vit" 
if not os.path.exists(path): os.makedirs(path)
with open(path+ "/logs.txt", 'a') as f:
        f.write(json.dumps(experiment_log) + '\n')