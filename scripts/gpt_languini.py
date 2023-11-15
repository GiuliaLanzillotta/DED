""" 

Author: Giulia Lanzillotta
Date created: Mon 13.11.23


We use the languini kitchen package (modified ad hog) to train a GPT model with distillation. 

"""


import os
import sys
import torch
import torch.multiprocessing as mp

from languini.train_lib import lm_trainer
from languini.train_lib import lr_schedules
from languini.common_lib import parallel_utils
from languini.common_lib import experiment_utils
from languini.dataset_lib import languini_books

from languini.common_lib.parallel_utils import mprint
from languini.common_lib.parallel_utils import LOCAL_RANK, WORLD_RANK, WORLD_SIZE

import languini.projects.gpt.configs as configs
from languini.projects.gpt.model import Model

TEACHER_CONFIG_NAME = "..." #TODO


def run(config, logger, teacher_config=None):
    c = config
    if teacher_config is None: teacher_config=c 
    
    mprint(f"{c.n_workers} workers detected. Using DistributedDataParallel. Local rank: {LOCAL_RANK}. Device: {c.device}")
    mprint(f"train batch size per worker/GPU: {c.train_batch_size // WORLD_SIZE}")
    mprint(f"eval batch size per worker/GPU: {c.eval_batch_size // WORLD_SIZE}")
    mprint(f"test batch size per worker/GPU: {c.test_batch_size // WORLD_SIZE}")
    mprint(f"gradient accumulation steps: {c.gradient_accumulation_steps}")

    mprint(f"WORLD_SIZE: {WORLD_SIZE}")  # total number of devices
    mprint(f"WORLD_RANK: {WORLD_RANK}")  # unique id within all devices
    mprint(f"LOCAL_RANK: {LOCAL_RANK}")  # unique id within the devices of this node

    mprint("Setup data sources ... ")
    # Compute the batch indices for this accelerator.
    assert c.train_batch_size % WORLD_SIZE == 0, "train batch size has to be a multiple of the number of workers"
    assert c.eval_batch_size % WORLD_SIZE == 0, "eval batch size has to be a multiple of the number of workers"
    train_batch_idxs = [i for i in range(c.train_batch_size) if i % WORLD_SIZE == WORLD_RANK]
    eval_batch_idxs = [i for i in range(c.eval_batch_size) if i % WORLD_SIZE == WORLD_RANK]

    END_OF_DOC_TOKEN = 2
    full_data_path = os.path.join(c.data_root, c.dataset)
    mprint(f"Loading data from {full_data_path}")
    train_ds = languini_books.LanguiniDatasetIterator(
        data_path=full_data_path,
        split='train',
        repeat=True,
        global_batch_size=c.train_batch_size,
        batch_idxs=train_batch_idxs,
        micro_batches=c.gradient_accumulation_steps,
        sequence_length=c.seq_len,
        device=c.device,
        end_of_doc_token=END_OF_DOC_TOKEN,
    )
    eval_ds = languini_books.LanguiniDatasetIterator(
        data_path=full_data_path,
        split='test',
        repeat=False,
        global_batch_size=c.eval_batch_size,
        batch_idxs=eval_batch_idxs,
        micro_batches=1,
        sequence_length=c.seq_len,
        device=c.device,
        end_of_doc_token=END_OF_DOC_TOKEN,
    )

    ## Setup Model
    mprint("Build model ... ")
    if WORLD_SIZE > 1:
        mprint("running on multiple devices ...")
    torch.manual_seed(c.seed)
    model = Model(config=c)
    #TODO: for now we use a teacher with the same architecture as the student. In principle, we can 
    # load any teacher architecture here
    teacher = Model(config=teacher_config)
    if c.compile != "None":
        model = torch.compile(model, mode=c.compile)
    if teacher_config.compile != "None":
        teacher = torch.compile(teacher, mode=teacher_config.compile)
    model = model.to(c.device)
    teacher = teacher.to(c.device) # teacher and student should be on the same device
    device_ids = [LOCAL_RANK] if c.device.type == "cuda" else None # must be None for non-cuda
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)  # we always use DDP so loading weights is simpler
    teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=device_ids) 
    ## Setup Optimiser
    opt = torch.optim.Adam(model.parameters(), lr=c.max_lr, betas=(0.9, 0.95), eps=1e-08)
    scheduler = lr_schedules.CosineLR(opt,
                                      warmup_steps=200,
                                      max_lr=c.max_lr,
                                      min_lr=c.min_lr,
                                      max_steps=c.decay_steps,
                                      decay_after=False)

    ## Setup Trainer
    trainer = lm_trainer.LMDistilTrainer(config=c,
                                        logger=logger,
                                        model=model,
                                        teacher=teacher,
                                        opt=opt,
                                        scheduler=scheduler,
                                        train_batches=train_ds,
                                        eval_batches=eval_ds)

    mprint("Begin training ... ")
    trainer.train()
    mprint("Done!")


def main():
    """Runs a Languini experiment using a GPT model."""

    # initialise distributed processes
    device = parallel_utils.init_distributed()
    mp.set_start_method("spawn")

    mprint("GPT Distil Languini Experiment")

    # parse the config name
    config_name = experiment_utils.parse_config_name(configs.config_names)
    mprint(f"Loading student config {config_name} and teacher config {TEACHER_CONFIG_NAME}")

    # load the config file
    config = configs.load_config(name=config_name)
    teacher_config = configs.load_config(name=TEACHER_CONFIG_NAME)
    project_path = os.path.dirname(os.path.abspath(__file__))
    mprint(f"project path: {project_path}")

    assert "checkpoint_path" in teacher_config.keys(), "Please provide a checkpoint for the teacher in the teacher config file."
    config.teacher_checkpoint_path = teacher_config.checkpoint_path

    # create parser and add custom args not extracted from the config
    parser = experiment_utils.create_parser_based_on_config(config)
    parser.add_argument("--compile", default="default", type=str, help=f"Which compile mode to use (None, default, reduce-overhead, max-autotune)")

    # parse args and make updates to the config
    args = parser.parse_args(sys.argv[2:])
    config = experiment_utils.update_config_given_args(config, args)
    config.project_path = project_path
    config.device = device
    
    # Check if the config matches the available hardware
    config = experiment_utils.check_hardware(config, world_size=WORLD_SIZE)

    # Generate experiment name based on config
    configs.add_exp_name(config)
    mprint(f"experiment name: {config.exp_name}")
    
    # Create the log folder, backup python files, and backup the hyperparameter config to a file
    logger = experiment_utils.setup_experiment(config)
    
    run(config, logger, teacher_config)


if __name__ == "__main__":
    main()
