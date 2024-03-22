# Giulia Lanzillotta . 17.10.2023
# Cifar 5M training experiment script 

"""

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
We use cifar5m, an extension to 5 mln images in order to train the student on more images than the teacher. 


example commands: 

python scripts/cifar5m.py --seed 11 --temper_labels --temperature 20 --alpha 1 --gpus_id 4 --buffer_size 12000 --distillation_type vanilla --batch_size 128  --checkpoints --notes cifar5m-distillation-CNN-TL --wandb_project DataEfficientDistillation
python scripts/cifar5m.py --seed 11 --fkd --alpha 0.003 --gpus_id 0 --buffer_size 12000 --distillation_type vanilla --batch_size 128  --checkpoints --notes cifar5m-distillation-resnet18 --wandb_project DataEfficientDistillation
python scripts/cifar5m.py --seed 11 --temperature 1 --alpha 1 --gpus_id 3 --buffer_size 12000 --lr 0.01 --optim_mom 0.8 --checkpoints  --teacher_network resnet18 --student_network resnet50 --batch_size 128 --checkpoints --notes cifar5m-distillation-resnet50-v2 --wandb_project DataEfficientDistillation
python scripts/cifar5m.py --seed 11 --temperature 20 --alpha 0 --gpus_id 4 --buffer_size 12000 --distillation_type vanilla --batch_size 128  --checkpoints --notes cifar5m-distillation-CNN-NTK --wandb_project DataEfficientDistillation
python scripts/cifar5m.py --teacher_network resnet18 --student_network resnet18 --alpha 0.9 --buffer_size 12000 --gpus_id 0 --temperature 20 --seed 11 --batch_size 128  --label_smoothing --checkpoints --notes cifar5m-resnet18-distillation-labelsmoothing --wandb_project DataEfficientDistillation

python scripts/cifar5m.py --teacher_network resnet18 --student_network resnet18 --alpha 0.9 --buffer_size 12000 --gpus_id 0 --temperature 20 --seed 11 --batch_size 128  --randomhead --checkpoints --notes cifar5m-resnet18-distillation-randomhead --wandb_project DataEfficientDistillation


Using hyperparameters from Torch recipe https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning 

Mobilenets: compact, no-residuals CNNs https://arxiv.org/pdf/1704.04861.pdf 

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
from utils.ntk import save_ntk
from utils.eval import evaluate, evaluate_CKA_teacher, evaluate_CKAandFA_teacher, evaluate_NTK_alignment_teacher, validation_and_agreement, distance_models, validation_agreement_function_distance
from dataset_utils.data_utils import load_dataset

try:
    import wandb
except ImportError:
    wandb = None

LOGITS_MAGNITUDE_TEACHER = 1.0 
AUGMENT = True
TEACHER_NETWORK = 'resnet18'
STUDENT_NETWORK = 'resnet18'

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
    path = base_path() + "/chkpts" + "/" + "cifar5m" + "/" + f"{STUDENT_NETWORK}/"
    if not os.path.exists(path): os.makedirs(path)
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(path+filename, path+'model_best.ckpt')

def load_checkpoint(device, best=False, filename='checkpoint.pth.tar', distributed=False):
    path = base_path() + "chkpts" + "/" + "cifar5m" + "/" + f"{TEACHER_NETWORK}/"
    if best: filepath = path + 'model_best.ckpt'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath, map_location=device)
          return checkpoint
    return None 

def parse_args(buffer=False):
    torch.set_num_threads(4)
    parser = ArgumentParser(description='script-experiment', allow_abbrev=False)
    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--checkpoints', action='store_true', help='Storing a checkpoint at every epoch. Loads a checkpoint if present.')
    parser.add_argument('--checkpoints_stud', action='store_true', help='Storing a checkpoint for the student.')
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
    parser.add_argument('--validate_subset', type=int, default=-1, help='If positive, allows validating on random subsets of the validation dataset during training.')
    parser.add_argument('--temperature', type=float, default=1., help='Temperature (prop to entropy) of the teacher outputs - only used with KL.')
    parser.add_argument('--temper_labels', action='store_true', help='Whether to use temperature in labels training (alpha>0)')
    parser.add_argument('--asymmetric_temperature', action='store_true', help='Temperature is only added to the teacher distribution.')
    parser.add_argument('--student_network', type=str, required=False, default="CNN", choices=["resnet18","CNN","mnet","resnet50"],help='Which network to use.')
    parser.add_argument('--teacher_network', type=str, required=False, default="CNN", choices=["resnet18","CNN"],help='Which network to use.')
    parser.add_argument('--label_smoothing', action='store_true', help='Using manually designed teacher through label smoothing.')
    parser.add_argument('--augmentations_off', action='store_true', help='Not using augmentations.')
    parser.add_argument('--randomhead', default=False, action='store_true',help='If provided, the teacher networks head is re-initialised to random.')


    add_management_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--alpha', type=float, default=0.5, required=True,
                        help='The weight of labels vs logits in the distillation loss (when alpha=1 only true labels are used)')
    parser.add_argument('--MSE', default=False, action='store_true',
                        help='If provided, the MSE loss is used for the student with labels .')
    parser.add_argument('--distillation_type', type=str, default='vanilla', choices=['vanilla', 'topK', 'inner', 'inner-parallel', 'topbottomK','randomK'],
                        help='Selects the distillation type, which determines the distillation loss.')
    parser.add_argument('--K', type=int, default=100, help='Number of activations to look at for *topK* distillation loss.')
    parser.add_argument('--C', type=int, default=20, help='Number of layer blocks to distill from. The layers are selected in a reverse ordering from the output to input.')
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

if args.augmentations_off:
        AUGMENT = False
        print("Not using augmentations")

C5m_train, C5m_test = load_dataset('cifar5m', augment=AUGMENT)

print(f"Randomly drawing {60000} samples for the Cifar5M base")
all_indices = set(range(len(C5m_train)))
random_indices = np.random.choice(list(all_indices), size=60000, replace=False)
teacher_data = Subset(C5m_train, random_indices)
C = args.C
STUDENT_NETWORK = args.student_network
TEACHER_NETWORK = args.teacher_network
# initialising the model
#teacher = CNN(c=C,num_classes=10,use_batch_norm=True) # adjusting for CIFAR 
if TEACHER_NETWORK=="resnet18": teacher = resnet18(num_classes=10)
elif TEACHER_NETWORK=="CNN": teacher = CNN(c=C, num_classes=10, use_batch_norm=True)
model_size = count_parameters(teacher)
print(f"Teacher {TEACHER_NETWORK} created with {model_size} parameters.")

setproctitle.setproctitle('{}_{}_{}'.format(f"{TEACHER_NETWORK}", args.buffer_size if 'buffer_size' in args else 0, "cifar5m"))

# start the training 
print(args)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(d) for d in args.gpus_id])
if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if args.wandb_name is None: 
                name = str.join("-",["offline", "cifar5m", f"{TEACHER_NETWORK}", args.conf_timestamp])
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



CHKPT_NAME = f'{TEACHER_NETWORK}-teacher.ckpt' # obtaineed with seed = 11

if not args.pretrained:
        teacher.train()
        optimizer, scheduler = setup_optimizerNscheduler(args, teacher)
        results = []
        best_acc = 0.
        start_epoch = 0
        epoch=0
        is_best = True


        if args.checkpoints: # resuming training from the last point
                checkpoint = load_checkpoint(device=device, best=False, filename=CHKPT_NAME, 
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
                results.append(val_acc)

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
        checkpoint = load_checkpoint(device=device, best=False, filename=CHKPT_NAME, distributed=not args.distributed=='no') #TODO: switch best off
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.to(device)
        final_val_acc_D = checkpoint['best_acc']

df = {'final_val_acc_D':final_val_acc_D}
wandb.log(df)


print(f"Randomly drawing {args.buffer_size} samples for the Cifar5M base")

if args.randomhead:     
        for (n,m) in teacher.named_children():
                if n == "fc":
                        print("Resetting ",n)
                        m.reset_parameters()


teacher.eval() # set the main model to evaluation

random_indices = np.random.choice(list(all_indices), 
                size=args.buffer_size, replace=False)

student_data = Subset(C5m_train, random_indices)
buffer_loader =  DataLoader(student_data, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=4,  
                            pin_memory=False)
#NOTE we keep the val loader of C100 for comparison
# val_loader = DataLoader(C5m_test, batch_size=args.batch_size, 
#                         shuffle=False, num_workers=3, pin_memory=True)


args = parse_args(buffer=True)
experiment_log = vars(args)
experiment_log['final_val_acc_D'] = final_val_acc_D
experiment_log['teacher_network'] = TEACHER_NETWORK
experiment_log['student_network'] = STUDENT_NETWORK
experiment_log['C'] = C



print("Starting student training ... ")
start = time.time()
# re-initialise model 
#student = CNN(c=C,num_classes=10,use_batch_norm=True) # adjusting for CIFAR 
if STUDENT_NETWORK=="resnet18": student = resnet18(num_classes=10)
elif STUDENT_NETWORK=="CNN": student = CNN(c=C, num_classes=10, use_batch_norm=True)
elif STUDENT_NETWORK=="resnet50": student = resnet50(num_classes=10)
model_size = count_parameters(student)
print(f"Student {STUDENT_NETWORK} created with {model_size} parameters.")

if args.distributed=='dp': 
      print(f"Parallelising buffer training on {len(args.gpus_id)} GPUs.")
      student = torch.nn.DataParallel(student, device_ids=args.gpus_id).to(device)
student.to(device)
student.train()

student_init = deepcopy(student)
student_init.eval() # only needed for evaluation purposes

optimizer, scheduler = setup_optimizerNscheduler(args, student, stud=True)


average_magnitude=0
alpha = args.alpha
loss_zero = False
T = args.temperature
a = 0.99 # label smoothing hp
# teacher_name = f'{TEACHER_NETWORK}-cifar5m'
student_name = f'{STUDENT_NETWORK}-{args.alpha}-{args.buffer_size}-{args.seed}-cifar5m'
ntk_savedir = path = base_path() + "/results" + "/" + "cifar5m" + "/" + f"{TEACHER_NETWORK}/" 
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
                if args.label_smoothing: 
                       if e==0: print("Label smoothing ON. ")
                       logits = F.one_hot(labels, num_classes=10).to(torch.float)*a
                       logits += (1-logits)*((1-a)/(1-10))
                
                optimizer.zero_grad()
                outputs = student(inputs)

                _, pred = torch.max(outputs.data, 1)
                _, pred_t = torch.max(logits.data, 1)
                correct += torch.sum(pred == labels).item()
                agreement += torch.sum(pred == pred_t).item()
                total += labels.shape[0]

                if LOGITS_MAGNITUDE_TEACHER == 1 and alpha>0: # estimate during the first epoch
                       average_non_max = (logits.sum(dim=1) - logits.max(dim=1)[0])/9 # average over the non-max outputs
                       average_magnitude += (logits.max(dim=1)[0] - average_non_max).sum(dim=0) 
                
                
                if args.MSE: 
                       labels_loss = F.mse_loss(outputs, F.one_hot(labels, num_classes=10).to(torch.float) * LOGITS_MAGNITUDE_TEACHER, reduction='none').mean(dim=1)  # Bobby's correction
                       logits_loss = F.mse_loss(outputs, logits, reduction='none').mean(dim=1)
                else:
                      teacher_p = F.softmax(logits/T, dim=1)
                      if args.asymmetric_temperature:
                             student_p = F.log_softmax(outputs, dim=1)
                             scaling_factor = 1
                      else: 
                             student_p=F.log_softmax(outputs/T, dim=1)
                             scaling_factor = T**2

                      if args.temper_labels: 
                             outputs = outputs/T #CHECK: no weighing on the labels?
                             labels_scaling_factor = 1
                      else: labels_scaling_factor = 1
                      labels_loss = F.cross_entropy(outputs, labels, reduction='none') * labels_scaling_factor
                      logits_loss = F.kl_div(input=student_p, target=teacher_p, log_target=False, reduction='none').sum(dim=1) * scaling_factor # temperature rescaling (for gradients)
                

                loss = alpha*labels_loss.mean() + (1-alpha)*logits_loss.mean()

                loss.backward()
                optimizer.step()

                assert not math.isnan(loss)
                progress_bar.prog(i, len(buffer_loader), e, 'Student', loss.item())
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
        if TEACHER_NETWORK==STUDENT_NETWORK: 
               teacher_student_distance = distance_models(teacher, student)
        else: teacher_student_distance=-1
        val_acc, val_agreement = validation_and_agreement(student, teacher, val_loader, 
                                                        device, num_samples=args.validate_subset)
        
        fa, cka = evaluate_CKAandFA_teacher(teacher, student, buffer_loader, device, batches=10)
        fa_init, cka_init = evaluate_CKAandFA_teacher(student_init, student, buffer_loader, device, batches=10)
        # ntk_cka = evaluate_NTK_alignment_teacher(teacher, student, buffer_loader, device, 
        #                                          num_batches=7, savedir=ntk_savedir, load=e>0)
        # ntk_cka = evaluate_NTK_alignment_teacher(teacher, student, ntk_savedir, buffer_loader, 
        #                         subset=5000, names=(teacher_name, student_name))

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
              #'epoch_ntk_alignment_train':ntk_cka
              #'epoch_fa_train':fa
        }
        

        wandb.log(df)
        

print("Training completed. Full evaluation and logging...")
end = time.time()

chkpt_filename=f'{STUDENT_NETWORK}-student-{args.seed}-{args.buffer_size}-{args.alpha}-{args.temperature}.ckpt'
if args.checkpoints_stud: 
                save_checkpoint({
                'epoch': e + 1,
                'state_dict': student.state_dict(),
                'best_acc': val_acc,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
                }, False, filename=chkpt_filename)



fa_train, cka_train = evaluate_CKAandFA_teacher(teacher, student, buffer_loader, device, batches=20)
fa_val, cka_val = evaluate_CKAandFA_teacher(teacher, student, val_loader, device, batches=10)


fa_train_init, cka_train_init = evaluate_CKAandFA_teacher(student_init, student, buffer_loader, device, batches=20)
fa_val_init, cka_val_init = evaluate_CKAandFA_teacher(student_init, student, val_loader, device, batches=10)
# ntk_cka_train = evaluate_NTK_alignment_teacher(teacher, student, buffer_loader, device, 
#                                                num_batches=7, savedir=ntk_savedir, load=True)
# ntk_cka_val = evaluate_NTK_alignment_teacher(teacher, student, val_loader, device, num_batches=7, 
#                                              savedir=None, load=False)


experiment_log['final_cka_train'] = cka_train
experiment_log['final_cka_val'] = cka_val
experiment_log['final_cka_train_init'] = cka_train_init
experiment_log['final_cka_val_init'] = cka_val_init
#experiment_log['final_ntk_train'] = ntk_cka_train
#experiment_log['final_ntk_val'] = ntk_cka_val
experiment_log['final_fa_train'] = fa_train
experiment_log['final_fa_val'] = fa_val
experiment_log['final_fa_train_init'] = fa_train_init
experiment_log['final_fa_val_init'] = fa_val_init
experiment_log['buffer_train_time'] = end-start
experiment_log['final_train_acc_S'] = train_acc


val_acc, val_agreement, val_function_distance = validation_agreement_function_distance(student, teacher, val_loader, device)
 

experiment_log['final_val_acc_S'] = val_acc
experiment_log['final_train_agreement'] = train_agreement
experiment_log['final_val_agreement'] = val_agreement
experiment_log['final_val_function_distance'] = val_function_distance
experiment_log['final_distance_teacher_student'] = teacher_student_distance

if not args.nowand:
        wandb.finish()


# dumping everything into a log file
path = base_path() + "results" + "/" + "cifar5m" + "/" + f"{STUDENT_NETWORK}_v2" 
if not os.path.exists(path): os.makedirs(path)
with open(path+ "/logs.txt", 'a') as f:
        f.write(json.dumps(experiment_log) + '\n')