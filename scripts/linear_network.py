# We investigate data efficiency of distillation in a simple case, where we specify the model and -importantly- the noise level. 
# Here, we use linear neural networks to model our data.
# date: 29.09.23 
# author: Giulia Lanzillotta 



"""commands to run this script 
python scripts/linear_network.py --gpus_id 0 --problem_type regression --D 500 --G 500 --N 10000  --C 1 --buffer_size 600 --noise 1.0 --alpha 0.0 --data_seed 13 --lr 0.00001  --batch_size 200 --n_epochs 20 --n_epochs_stud 20 --notes linearnet-script --wandb_project DataEfficientDistillation


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

import torch
import torch.nn.functional as F

internal_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(internal_path)
sys.path.append(internal_path + '/datasets')
sys.path.append(internal_path + '/utils')

import shutil
from utils.args import add_management_args, add_rehearsal_args
from utils.conf import set_random_seed, get_device, base_path
from utils.status import ProgressBar
from utils.stil_losses import *
from utils.nets import *
from utils.eval import evaluate, validation_and_agreement, distance_models, evaluate_regression, evaluate_classification


from sklearn.datasets import *
from sklearn import linear_model

try:
    import wandb
except ImportError:
    wandb = None


N_TEST = 1000

def save_checkpoint(state, is_best, problem_type, filename='checkpoint.pth.tar'):
    path = base_path() + "/chkpts" + "/" + problem_type + "/" + "linearnet/"
    if not os.path.exists(path): os.makedirs(path)
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(path+filename, path+'model_best.pth.tar')

def load_checkpoint(problem_type, best=False, filename='checkpoint.pth.tar'):
    path = base_path() + "/chkpts" + "/" + problem_type + "/" + "linearnet/"
    if best: filepath = path + 'model_best.pth.tar'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath)
          return checkpoint
    return None 


def parse_args():
    parser = ArgumentParser(description='linear-experiment', allow_abbrev=False)
    parser.add_argument('--problem_type', type=str, default='regression', choices=['regression','classification','clutering'],
                        help="the type of problem to be solved")
    parser.add_argument('--N', type=int, default=10000, help="Dataset size")
    parser.add_argument('--buffer_size', type=int, default=100, help="(Random) data subset size")
    parser.add_argument('--D', type=int, default=100, help="Number of input features")
    parser.add_argument('--G', type=int, default=100, help="Number of predictive features.")
    parser.add_argument('--C', type=int, default=1, help="Number of targets")
    parser.add_argument('--effective_rank', type=int, default=None, help="Effective rank of the input matrix")
    parser.add_argument('--noise', type=float, default=1.0, help="sd of the Gaussian noise added to the output")
    parser.add_argument('--label_noise', type=float, default=0.0, help="percentage of label flips in classification task")
    parser.add_argument('--data_seed', type=int, help="seed used to generate the data")
    parser.add_argument('--alpha', type=float, default=0.5, required=True,
                        help='The weight of labels vs logits in the distillation loss (when alpha=1 only true labels are used)')
    parser.add_argument('--distillation_type', type=str, default='vanilla', choices=['vanilla','hard_targets'],
                        help="distillation mechanism to use ... see the code for details")
    
    # network optimization parameters ------------
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--checkpoints', action='store_true', help='Storing a checkpoint at every epoch. Loads a checkpoint if present.')
    parser.add_argument('--pretrained', action='store_true', help='Using a pre-trained network instead of training one.')
    parser.add_argument('--optim_wd', type=float, default=0, help='optimizer weight decay.')
    parser.add_argument('--optim_adam', default=False, action='store_true', help='Using the Adam optimizer instead of SGD.')
    parser.add_argument('--optim_mom', type=float, default=0, help='optimizer momentum.')
    parser.add_argument('--optim_warmup', type=int, default=0, help='Number of warmup epochs.')
    parser.add_argument('--optim_nesterov', type=int, default=0, help='optimizer nesterov momentum.')
    parser.add_argument('--optim_cosineanneal', default=False, action='store_true', help='Enabling cosine annealing of learning rate..')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--n_epochs_stud', type=int, default=10, help='Number of student epochs.')
    parser.add_argument('--batch_size', type=int, default = 100, help='Batch size.')
    parser.add_argument('--MSE', default=False, action='store_true',
                        help='If provided, the MSE loss is used for the student with labels .')
    
    
    add_management_args(parser)
    args = parser.parse_args()
    return args 




# - parse args 
args = parse_args()
args.conf_jobnum = str(uuid.uuid4())
args.conf_timestamp = str(datetime.datetime.now())
args.conf_host = socket.gethostname()

if args.seed is not None:
        set_random_seed(args.seed)
print(args)

setproctitle.setproctitle('{}_{}_{}'.format(args.problem_type, args.buffer_size, "linearnet"))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(d) for d in args.gpus_id])
device = get_device([0]) # returns the first device in the list


experiment_log = vars(args)
# start the wandb server
if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if args.wandb_name is None: 
                name = str.join("-",["offline", "linearnet", args.problem_type, args.conf_timestamp])
        else: name = args.wandb_name
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
                        name=name, notes=args.notes, config=vars(args)) 
        args.wandb_url = wandb.run.get_url()
print(file=sys.stderr)
# - load get dataset 
# we use data generators from scikit : https://scikit-learn.org/stable/datasets/sample_generators.html#sample-generators
#TODO: maybe store the data somewhere and load it?
print("Creating dataset ... ")
if args.problem_type=='regression':
        # make_regression produces regression targets as an optionally-sparse 
        # random linear combination of random features, with noise. 
        # Its informative features may be uncorrelated, or low rank 
        # (few features account for most of the variance).
        X, Y, theta_star = make_regression(n_samples=args.N+N_TEST, n_features=args.D, 
                                n_informative=args.G, n_targets=args.C, bias=0.0, 
                                effective_rank=args.effective_rank, noise=args.noise, 
                                shuffle=True, coef=True, random_state=args.data_seed)
elif args.problem_type=='classification':
       X, Y = make_blobs(n_samples=args.N+N_TEST, n_features=args.D, centers=args.C, 
                         cluster_std=args.noise, random_state=args.data_seed, center_box=(0,1))
       theta_star = np.zeros(args.D)

X = torch.Tensor(X).to(device)
Y = torch.Tensor(Y).to(device)
if args.problem_type=='classification': Y = Y.type(torch.LongTensor) 
# dividing in train and test sets 
X_train = X[:-N_TEST,:]
Y_train = Y[:-N_TEST]
X_test = X[-N_TEST:,:]
Y_test = Y[-N_TEST:]

print("..done") 
eval_fun = lambda x : 0 #dummy placeholder
if args.problem_type=='regression':     
        eval_fun = evaluate_regression
elif args.problem_type=='classification':
       eval_fun = evaluate_classification

# - get the teacher model
teacher = LinearNet(dim_in=args.D, dim_out=args.C).to(device)
progress_bar = ProgressBar(verbose=not args.non_verbose)
if not args.pretrained:
       print("Training teacher model...")
       teacher.train()
       optimizer = torch.optim.SGD(teacher.parameters(), 
                                   lr=args.lr, 
                                   weight_decay=args.optim_wd, 
                                   momentum=args.optim_mom)

       if args.problem_type=='classification': best_acc = 0.
       else: best_acc = 10e14 # arbitrarily high number
       start_epoch = 0
       if args.checkpoints: # resuming training from the last point
                checkpoint = load_checkpoint(problem_type=args.problem_type, best=False) #TODO: switch best off
                if checkpoint: 
                        teacher.load_state_dict(checkpoint['state_dict'])
                        teacher.to(device)
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        start_epoch = checkpoint['epoch']
                        best_acc = checkpoint['best_acc']

       for epoch in range(start_epoch, args.n_epochs):
                avg_loss = 0.0
                correct, total = 0.0, 0.0
                for i in range(0, args.N, args.batch_size):
                        if args.debug_mode and i > 3: # only 3 batches in debug mode
                                break
                        upper_bound = min(i+args.batch_size,X.shape[0]-1)
                        inputs = X_train[i:upper_bound,:]
                        labels = Y_train[i:upper_bound]
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = teacher(inputs)
                        if args.problem_type=='classification':
                                _, pred = torch.max(outputs.data, 1)
                                correct += torch.sum(pred == labels).item()
                                total += labels.shape[0]
                                loss = F.cross_entropy(outputs, labels) #TODO: maybe MSE?
                        else: 
                                loss = F.mse_loss(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        assert not math.isnan(loss)
                        progress_bar.prog(i, X.shape[0], epoch, 'D', loss.item())
                        avg_loss += loss

                with torch.no_grad():
                        if args.problem_type=='classification': train_acc = correct/total * 100
                        else: train_acc = eval_fun(teacher, X_train, Y_train)
                        val_acc = eval_fun(teacher, X_test, Y_test)

                # best val accuracy -> selection bias on the validation set
                if args.problem_type=='classification': is_best = val_acc > best_acc
                else: is_best = val_acc < best_acc
                best_acc = max(val_acc, best_acc)

                print('\Train accuracy : {} %'.format(round(train_acc, 2)), file=sys.stderr)
                print('\Val accuracy : {} %'.format(round(val_acc, 2)), file=sys.stderr)
                
                df = {'epoch_loss_D':avg_loss/X.shape[0],
                'epoch_train_acc_D':train_acc,
                'epoch_val_acc_D':val_acc}
                wandb.log(df)


                if args.checkpoints: 
                        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': teacher.state_dict(),
                        'best_acc': best_acc,
                        'optimizer' : optimizer.state_dict()
                        }, is_best, problem_type=args.problem_type)

else: 
        checkpoint = load_checkpoint(problem_type=args.problem_type, best=False) #TODO: switch best off
        teacher.load_state_dict(checkpoint['state_dict'])
        teacher.to(device)

teacher.eval()
teacher_val_accuracy = eval_fun(teacher, X_test, Y_test)
experiment_log['final_val_acc_D'] = teacher_val_accuracy
print("...done")
# start the training 


# create the student dataset 
all_indices = set(range(args.N)) #note: no test indices here
random_indices = np.random.choice(list(all_indices), size=args.buffer_size, replace=False)
left_out_indices = list(all_indices.difference(set(random_indices.flatten())))
# buffer data
X_buffer = X[random_indices]
Y_buffer = Y[random_indices]
X_left_out = X[left_out_indices]
Y_left_out = Y[left_out_indices]


print("Starting buffer training ... ")
start = time.time()
# initialise student model 
student = LinearNet(dim_in=args.D, dim_out=args.C).to(device)
student.train()
optimizer = torch.optim.SGD(student.parameters(), 
                            lr=args.lr, 
                            weight_decay=args.optim_wd, 
                            momentum=args.optim_mom)
# set the targets to use
progress_bar = ProgressBar(verbose=not args.non_verbose)

alpha = args.alpha
for e in range(args.n_epochs_stud):
        if args.debug_mode and e > 3: # only 3 batches in debug mode
                break
        avg_loss = 0.0
        correct, agreement = 0.0, 0.0
        for i in range(0, args.buffer_size, args.batch_size):
                if args.debug_mode and i > 3: # only 3 batches in debug mode
                        break
                upper_bound = min(i+args.batch_size,X.shape[0]-1)
                inputs = X_buffer[i:upper_bound,:]
                labels = Y_buffer[i:upper_bound]
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad(): logits = teacher(inputs)
                optimizer.zero_grad()
                outputs = student(inputs)
                
                # the distillation loss
                logits_loss = vanilla_distillation(outputs, logits)
                # the labels loss 
                if args.problem_type=='classification':
                        if args.MSE: labels_loss = F.mse_loss(outputs, F.one_hot(labels, num_classes=args.C).to(torch.float))  # Bobby's correction
                        else: labels_loss = F.cross_entropy(outputs, labels)
                else: labels_loss = F.mse_loss(outputs, labels) 
                loss = alpha*labels_loss + (1-alpha)*logits_loss
                loss.backward()
                optimizer.step()
                
                if args.problem_type=='classification':
                        _, pred = torch.max(outputs.data, 1)
                        _, pred_t = torch.max(logits.data, 1)
                        correct += torch.sum(pred == labels).item()
                        agreement += torch.sum(pred == pred_t).item()
                else: agreement += labels_loss

                assert not math.isnan(loss)
                progress_bar.prog(i, args.buffer_size, e, 'S', loss.item())
                avg_loss += loss

        avg_loss = avg_loss/args.buffer_size
                
        if args.problem_type=='classification': 
                train_acc = correct/args.buffer_size * 100
                train_agreement = (agreement/args.buffer_size) * 100
        else: 
               train_acc = eval_fun(student, X_buffer, Y_buffer)
               train_agreement = (agreement/args.buffer_size)

        train_leftout_acc = eval_fun(student, X_left_out, Y_left_out)
        teacher_student_distance = distance_models(teacher, student)
        val_acc = eval_fun(student, X_test, Y_test)


        print('\nTrain accuracy : {} %'.format(round(train_acc, 2)), file=sys.stderr)
        print('Train left-out accuracy : {} %'.format(round(train_leftout_acc, 2)), file=sys.stderr)
        print('\Val accuracy : {} %'.format(round(val_acc, 2)), file=sys.stderr)
        
        df = {'epoch_loss_S':avg_loss,
              'epoch_train_acc_S':train_acc,
              'epoch_train_agreement':train_agreement,
              'epoch_distance_teacher_student':teacher_student_distance,
              'epoch_train_leftout_acc_S':train_leftout_acc,
              'epoch_val_acc_S':val_acc}
        wandb.log(df)


print("Training completed. Full evaluation and logging...")
end = time.time()

end = time.time()
print("...done")

experiment_log['buffer_train_time'] = end-start
experiment_log['final_train_acc_S'] = train_acc
experiment_log['final_train_leftout_acc_S'] = train_leftout_acc
experiment_log['final_val_acc_S'] = val_acc
experiment_log['final_distance_teacher_student'] = teacher_student_distance


if not args.nowand:
        wandb.log(experiment_log)
        wandb.finish()


# dumping everything into a log file
path = base_path() + "results" + "/" + args.problem_type + "/" + "linearnet" 
if not os.path.exists(path): os.makedirs(path)
with open(path+ "/logs.txt", 'a') as f:
        f.write(json.dumps(experiment_log) + '\n')