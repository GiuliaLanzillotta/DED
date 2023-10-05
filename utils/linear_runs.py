"""Takes a job run command and prints out a sequence of commands with multiple seeds 
dividing the GPUs equally between them"""
import os
import sys
from copy import copy
import subprocess


SEEDS = [11, 13, 21, 33, 55 ,5,138,228,196,118]
BUFFER_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
NOISE = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
PARALLEL_ORDER = 5

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
    for alpha in [0.0, 1.0]:
        for seed in SEEDS:
            #for k in K: # for topK distillation
            for sigma in NOISE: # inner block distillation
                new_argv = copy(sys.argv)
                new_argv.append(f'--buffer_size {buf_size} ')
                new_argv.append(f'--seed {seed} ')
                new_argv.append(f'--alpha {alpha}')
                new_argv.append(f'--noise {sigma}')
                job_count+=1
                # gpu_count=next_gpu
                all_commands.append(" ".join(new_argv[1:]))
                if job_count==PARALLEL_ORDER:
                    subprocess.run(["utils/run_multiple_commands.sh"]+all_commands)
                    all_commands=[]
                    job_count=0 
