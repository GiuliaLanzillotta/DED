"""Takes a job run command and prints out a sequence of commands with multiple seeds 
dividing the GPUs equally between them"""
import os
import sys
from copy import copy
import subprocess

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')
sys.path.append(mammoth_path + '/utils')

SEEDS = [11,13,21,33,55]#,5,138,228,196,118]#[11,13]#
BUFFER_SIZES = [60000]#[480000]#, 480000]#[60000]#[360000, 480000] #1200, 12000, 60000
K = [10, 50, 100, 200, 500]
#NUM_GPUS_PER_COMMAND = 2 
PARALLEL_ORDER = 4
GPUIDS = [0, 1, 2, 3]

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

for buf_size in BUFFER_SIZES: 
    for alpha in [0.0]:
        for seed in SEEDS:
            for k in K: # for topK distillation
                new_argv = copy(sys.argv)
                new_argv.append(f'--buffer_size {buf_size} ')
                new_argv.append(f'--seed {seed} ')
                new_argv.append(f'--alpha {alpha}')
                new_argv.append(f'--K {k}')
                new_argv.append(f'--gpus_id {GPUIDS[job_count]}')
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


# python utils/multiply_and_run_commands.py python scripts/imagenet.py  --validate_subset 2000 --batch_size 64 --checkpoints --MSE --notes imagenet-script-all-exp --wandb_project DataEfficientDistillation
