"""Takes a job run command and prints out a sequence of commands with multiple seeds 
dividing the GPUs equally between them

example commands: 

python utils/cifar5m_runs.py python scripts/cifar5m.py --distillation_type vanilla --batch_size 128 --checkpoints --notes cifar5m-distillation --wandb_project DataEfficientDistillation

python utils/cifar5m_runs.py python scripts/cifar10_mixed.py --reset_optim --teacher_off --distil_proportion 0.4 --batch_size 128  --checkpoints --notes cifar10-mixeddistillation --wandb_project DataEfficientDistillation

"""
import os
import sys
from copy import copy
import subprocess


SEEDS = [11, 13, 21, 33, 55]#,5,138,228,196,118
BUFFER_SIZES = [600, 1200, 6000, 12000, 24000] # 1200000, 600000, 120000, 60000
PARALLEL_ORDER = 5
GPUIDS = [2, 3, 4, 5, 6]

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

for alpha in [0.0, 1.0]:
    for seed in SEEDS:
        #for k in K: # for topK distillation
        #for b in N_BLOCKS: # inner block distillation
            new_argv = copy(sys.argv)
            new_argv.append(f'--seed {seed} ')
            if alpha==0:
                new_argv.append(f'--teacher_off')
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
