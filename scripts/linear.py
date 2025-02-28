# We investigate data efficiency of distillation in a simple linear case, where we specify the model and -importantly- the noise level. 
# date: 29.09.23 
# author: iulia Lanzillotta 



"""commands to run this script 
python scripts/linear.py --problem_type regression --D 500 --N 10000 --buffer_size 100 --noise 1.0 --alpha 1.0 --data_seed 13
python utils/linear_runs.py python scripts/linear.py --D 500 --G 300 --N 10000 --data_seed 13 --nowand 1 --notes linear-regression-distillation --wandb_project DataEfficientDistillation


python scripts/linear.py --problem_type classification --nowand 1 --D 500 --N 10000  --C 2 --buffer_size 1000 --noise 10.0 --data_seed 13 --distillation_type hard_targets
python utils/linear_runs.py python scripts/linear.py --problem_type classification --nowand 1 --D 500 --N 10000  --C 2 --data_seed 13 --distillation_type hard_targets


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

experiment_log = vars(args)
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
print("Fitting teacher model...")
setproctitle.setproctitle('{}_{}_{}'.format(args.problem_type, args.buffer_size, "linear"))
if args.problem_type=='regression':
        teacher = linear_model.LinearRegression()
elif args.problem_type=='classification':
       teacher = linear_model.LogisticRegression(random_state=args.seed, 
                                                 max_iter=200, penalty='none', 
                                                 multi_class='multinomial')
teacher.fit(X_train, Y_train)
teacher_val_accuracy = eval_fun(teacher, X_test, Y_test)
theta_teacher = teacher.coef_
teacher_optimum_distance = np.linalg.norm(theta_star-theta_teacher)

experiment_log['final_val_acc_D'] = teacher_val_accuracy
experiment_log['final_distance_teacher_optimum'] = teacher_optimum_distance

print("...done")
# start the training 

# start the wandb server
if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if args.wandb_name is None: 
                name = str.join("-",["offline", "linear", args.problem_type, args.conf_timestamp])
        else: name = args.wandb_name
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
                        name=name, notes=args.notes, config=vars(args)) 
        args.wandb_url = wandb.run.get_url()
print(file=sys.stderr)

# create the student dataset 
all_indices = set(range(args.N))
random_indices = np.random.choice(list(all_indices), size=args.buffer_size, replace=False)
left_out_indices = list(all_indices.difference(set(random_indices.flatten())))
# buffer loader
X_buffer = X[random_indices]
Y_buffer = Y[random_indices]
X_left_out = X[left_out_indices]
Y_left_out = Y[left_out_indices]


print("Starting buffer training ... ")
start = time.time()
# initialise student model 
if args.problem_type=='regression' or (args.distillation_type=='vanilla' and args.alpha==0):
        student = linear_model.LinearRegression() #note here 
else: student = linear_model.LogisticRegression(random_state=args.seed, 
                                                 max_iter=200, penalty='none', 
                                                 multi_class='multinomial')
# set the targets to use
labels = Y_buffer
teacher_predictions = 0
if args.alpha < 1:
        if args.problem_type=='regression' or args.distillation_type=='hard_targets':
                teacher_predictions = teacher.predict(X_buffer)
        elif args.problem_type=='classification':  
                teacher_predictions = teacher.predict_proba(X_buffer)
                labels = F.one_hot(torch.tensor(Y_buffer), num_classes=args.C).detach().to(float).numpy()
targets = args.alpha * labels + (1-args.alpha)*teacher_predictions

       
# train the student model 
student.fit(X_buffer, targets)
student_train_accuracy = eval_fun(student, X_buffer, Y_buffer)
student_val_accuracy = eval_fun(student, X_test, Y_test)
student_left_out_accuracy = eval_fun(student, X_left_out, Y_left_out)
teacher_student_distance = np.linalg.norm(student.coef_-theta_teacher)
student_optimum_distance = np.linalg.norm(student.coef_-theta_star)

end = time.time()
print("...done")
experiment_log['buffer_train_time'] = end-start
experiment_log['final_train_acc_S'] = student_train_accuracy
experiment_log['final_train_leftout_acc_S'] = student_left_out_accuracy
experiment_log['final_val_acc_S'] = student_val_accuracy
experiment_log['final_distance_teacher_student'] = teacher_student_distance
experiment_log['final_distance_student_optimum'] = student_optimum_distance




if not args.nowand:
        wandb.log(experiment_log)
        wandb.finish()


# dumping everything into a log file
path = base_path() + "results" + "/" + args.problem_type + "/" + "linear" 
if not os.path.exists(path): os.makedirs(path)
with open(path+ "/logs.txt", 'a') as f:
        f.write(json.dumps(experiment_log) + '\n')