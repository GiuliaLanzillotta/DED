"""Takes a job run command and prints out a sequence of commands with multiple seeds 
dividing the GPUs equally between them

example commands: 

python utils/logistic_runs.py python scripts/logistic_regression.py  --batch_size 256  --dataset cifar100 --network resnet18
python utils/logistic_runs.py python scripts/logistic_regression.py  --batch_size 256  --dataset cifar5m --network CNN


"""
import os
import sys
from copy import copy
import subprocess

SEEDS = [11, 13, 21, 33, 55]#
BUFFER_SIZES = [1200, 6000, 12000, 24000, 48000, 60000]
#BUFFER_SIZES = [1200, 6000, 12000, 24000, 48000, 60000, 120000, 600000]
TEMPERATURES = [0.1, 1, 3, 5, 10, 20]

PARALLEL_ORDER = 4
#GPUIDS = [0,1,2,3,4,5,6,7]
GPUIDS = [0,1]
def crange(start, end, modulo):
    # implementing circular range
    if start > end:
        while start < modulo:
            yield start
            start += 1
        start = 0

    while start < end:
        yield start
        start += 1


all_commands=[]
gpu_count=0
job_count=0

for b in BUFFER_SIZES:
    for seed in SEEDS:
        for alpha in [0.0, 1.0]:
            for T in TEMPERATURES:
            #for k in K: # for topK distillation
            #for b in N_BLOCKS: # inner block distillation
                
                if alpha==1.0 and T!=1:
                   continue
                
                new_argv = copy(sys.argv)
                
                new_argv.append(f'--seed {seed} ')
                
                # else:
                #T=20
                #new_argv.append(f'--inner_temperature {T}')
                #new_argv.append(f'--conditional_teacher')
                new_argv.append(f'--alpha {alpha} ')
                    #new_argv.append(f'--beta {beta} ')
                new_argv.append(f'--buffer_size {b} ')
                if T==10000:
                   new_argv.append(f'--MSE')
                else: 
                    new_argv.append(f'--temperature {T} ')
                #if seed==11: 
                #       new_argv.append('--checkpoints_stud')
                #if alpha==1:
                #    new_argv.append(f'--teacher_off')
                #new_argv.append(f"--distil_proportion {p}")
                #new_argv.append(f'--N_BLOCKS {b}')
                #new_argv.append(f'--K {k}')
                gpu_idx = job_count % len(GPUIDS)
                new_argv.append(f'--gpus_id {GPUIDS[gpu_idx]}')
                # next_gpu = (gpu_count+NUM_GPUS_PER_COMMAND)%(len(GPUIDS))
                # new_argv.append('--gpus_id '+ \
                #     " ".join([str(GPUIDS[c]) for c in \
                #     crange(gpu_count,next_gpu,len(GPUIDS))])) 
                job_count+=1
                # gpu_count=next_gpu
                all_commands.append(" ".join(new_argv[1:]))
                if job_count==PARALLEL_ORDER:
                    subprocess.run(["utils/run_multiple_commands.sh"]+all_commands)
                    all_commands=[]
                    job_count=0 

if len(all_commands)>0:
     # sending the last jobs in the queue
     subprocess.run(["utils/run_multiple_commands.sh"]+all_commands)
