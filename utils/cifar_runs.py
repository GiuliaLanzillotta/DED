"""Takes a job run command and prints out a sequence of commands with multiple seeds 
dividing the GPUs equally between them

example commands: 


python utils/cifar_runs.py python scripts/cifar5m.py  --teacher_network CNN --student_network resnet18 --batch_size 128 --checkpoints --notes cifar5m-distillation-resnet18-v2 --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar5m_zeroloss.py  --MSE --distillation_type vanilla --batch_size 128 --checkpoints --notes cifar5m-distillation-zero_loss --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar5m_SSL.py --distillation_type vanilla --batch_size 128 --checkpoints --notes cifar5m-distillation-SSL --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar10_mixed.py --reset_optim --batch_size 128  --checkpoints --notes cifar10-mixeddistillation-all --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar5m_small.py --distillation_type vanilla --batch_size 128  --checkpoints --notes cifar5msmall-distillation --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar5m_recurrent.py --distillation_type vanilla --batch_size 128  --checkpoints --notes cifar5mrecurrent-distillation --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar5m_2heads.py --MSE --distillation_type vanilla --batch_size 128  --checkpoints --notes cifar5m-distillation-2heads_big --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar5m_conditionalteacher.py --MSE --K 3 --batch_size 128  --checkpoints --notes cifar5m-convnet_scaledloss-distillation --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar100.py --batch_size 128  --checkpoints --notes cifar100-resnet18-distillation-v2 --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar100_kerd.py  --lamdafk 1  --cka  --distillation_type vanilla   --checkpoints --notes cifar100-resnet18-distillation-CKA --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar5m.py --temper_labels --alpha 1 --distillation_type vanilla --batch_size 128  --checkpoints --notes cifar5m-distillation-CNN-TL --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar5m.py --batch_size 128  --checkpoints --notes cifar5m-distillation-CNN --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar100_kerd.py --block_gradient --symmetric --outer_temperature 1 --lamdafk 100 --lamdafr 0.01  --cka --teacher_targets   --distillation_type vanilla   --checkpoints --notes cifar100-resnet18-distillation-CKA --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar100_linear_retraining.py  --distillation_type vanilla --batch_size 128  --checkpoints --notes cifar100-resnet18-linretraining --wandb_project DataEfficientDistillation
python utils/cifar_runs.py python scripts/cifar100.py --batch_size 128  --label_smoothing --checkpoints --notes cifar100-resnet18-distillation-labelsmoothing --wandb_project DataEfficientDistillation


"""
import os
import sys
from copy import copy
import subprocess


SEEDS = [11, 13, 21, 33, 55]#,5,138,228,196,118
#SEEDS=[11]
#BUFFER_SIZES = [90000, 120000, 200000, 600000] 
#BUFFER_SIZES = [1200, 6000, 12000, 24000, 48000]
BUFFER_SIZES = [1200, 6000, 12000, 24000, 48000, 60000, 90000, 120000, 200000, 600000]
#BUFFER_SIZES = [1200, 6000, 12000, 24000, 48000, 60000]
TEMPERATURES = [0.1, 1, 3, 5, 10, 20, 100, 10000]
#TEMPERATURES = [0.1, 0.33, 0.5, 1.0, 2.0]

PARALLEL_ORDER = 4
#GPUIDS = [0,1,2,3,4,5,6,7]
GPUIDS = [4,5,6,7]
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
                # if seed in [11,13,33]: 
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
