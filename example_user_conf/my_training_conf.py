from omegaconf import MISSING

###################################################################################################
############################################ Base conf ############################################
###################################################################################################
# These are generic classes that need full instantiation
from GaussianProxy.conf.training_conf import (
    Accelerate,
    AccelerateLaunchArgs,
    Checkpointing,
    Config,
    DataLoader,
    DDIMSchedulerConfig,
    Evaluation,
    InvertedRegeneration,
    MetricsComputation,
    OneCycleLRConfig,
    SimilarityWithTrainData,
    SimpleGeneration,
    Slurm,
    Training,
)

###################################################################################################
########################################## Defaults conf ##########################################
###################################################################################################
defaults = [
    {"dataset": "path/to/dataset_conf"},
    "hydra/job_logging/custom",
    "_self_",
]
# fmt: off

# ------------------------------------------- Job launch ------------------------------------------
slurm = Slurm(
    enabled         = False,
    monitor         = False,
    account         = "icr@a100",
    partition       = "a100",
    constraint      = "a100",
    qos             = "dev",
    nodes           = 1,
    num_gpus        = 8,
    max_num_requeue = 3,
    output_folder   = "${hydra:run.dir}",
    email           = "tboyer@bio.ens.psl.eu",
)

accelerate_launch_args = AccelerateLaunchArgs(
    machine_rank      = 0,
    num_machines      = 1,
    gpu_ids           = "all",
    rdzv_backend      = "static",
    same_network      = "true",
    mixed_precision   = "bf16",
    num_processes     = 4,
    main_process_port = 29502,
    dynamo_backend    = "inductor",
)

accelerate = Accelerate(
    launch_args = accelerate_launch_args,
    offline     = False,  # TODO: move this arg that does not belong here (make it general like debug)
)

# ---------------------------------------------- Data ---------------------------------------------
data_loader = DataLoader(
    num_workers           = 6,
    train_prefetch_factor = 4,
    pin_memory            = True,
    persistent_workers    = True,
)

# -------------------------------------------- Training -------------------------------------------
training = Training(
    gradient_accumulation_steps = 1,
    train_batch_size            = 32,
    max_grad_norm               = 1,
    nb_time_samplings           = 200_000,
    unpaired_data               = False,
    reweight_sampling           = True,
)

checkpointing = Checkpointing(
    checkpoints_total_limit  = 15,
    resume_from_checkpoint   = True,
    checkpoint_every_n_steps = 5000,
    chckpt_base_path         = MISSING,
)

lr_scheduler = OneCycleLRConfig(
    max_lr = 1e-4,
)

# ------------------------------------------- Evaluation ------------------------------------------
# naming convention is lowercase + underscore; has to be respected for debug args modification
metrics_compute = MetricsComputation(
    nb_samples_to_gen_per_time        = 10_000,
    batch_size                        = 256,
    nb_diffusion_timesteps            = 50,
    selected_times                    = ["timestep_0", "timestep_2", "timestep_3"],
    augmentations_for_metrics_comp    = [],
    also_compute_metrics_on_all_times = False,
)

simple_generation = SimpleGeneration(
    nb_diffusion_timesteps = 50,
    n_rows_displayed       = 4,  # TODO: merge training & evaluation configs
    nb_generated_samples   = 16,  # TODO: merge training & evaluation configs
)

inverted_regeneration = InvertedRegeneration(
    nb_diffusion_timesteps           = 50,
    nb_inversion_diffusion_timesteps = 100,
    n_rows_displayed                 = 8,  # TODO: not used in training!
    nb_generated_samples             = 16,
    nb_video_times_in_parallel       = 4,  # TODO: not used in training!
    nb_video_timesteps               = 50,
    gen_times_type                   = ("evenly_spaced_from_inversion", 0.95),
)

sim_with_train = SimilarityWithTrainData( # must be put after metrics_compute!
    nb_generated_samples   = -1,  # TODO: not used
    batch_size             = 4096,
    nb_batches_shown       = -1,  # TODO: not used
    n_rows_displayed       = -1,  # TODO: not used
    nb_diffusion_timesteps = -1,  # TODO: not used
)

evaluation = Evaluation(
    every_n_opt_steps      = 20_000,
    batch_size             = 16,  # TODO: remove this and use config from above
    nb_video_timesteps     = 50,  # TODO: remove this and use config from above
    strategies             = [simple_generation, inverted_regeneration, metrics_compute, sim_with_train],
)

# ------------------------------------------- Diffusion -------------------------------------------
dynamic = DDIMSchedulerConfig(
    num_train_timesteps    = 3000,
    clip_sample            = False,
    clip_sample_range      = 1,
    thresholding           = True,
    sample_max_value       = 1,
    prediction_type        = "v_prediction",
    rescale_betas_zero_snr = False,
    timestep_spacing       = "leading",
)

# ---------------------------------------------- Model --------------------------------------------
from my_conf.net.net_256_3_20M import net, time_encoder  # noqa: E402

# ------------------------------------------ Final Config -----------------------------------------
config = Config(
    # defaults
    defaults                      = defaults,
    # model
    dynamic                       = dynamic,
    net                           = net,
    time_encoder                  = time_encoder,
    # script
    launcher_script_parent_folder = "path/to/launcher_script_parent_folder",
    script                        = "train",
    # experiment variables
    exp_parent_folder             = "path/to/experiments_parent_folder",
    project                       = MISSING,
    run_name                      = MISSING,
    # hydra
    hydra                         = {"run": {"dir": "${exp_parent_folder}/${project}/${run_name}"}},
    # slurm
    slurm                         = slurm,
    # accelerate
    accelerate                    = accelerate,
    # misc.
    debug                         = False,
    profile                       = False,
    # experiment tracker
    logger                        = "wandb",
    entity                        = "wandb_entity_name",
    resume_method                 = "rewind",
    # checkpointing
    checkpointing                 = checkpointing,
    # dataset
    dataset                       = MISSING,
    # dataloaders
    dataloaders                   = data_loader,
    # training
    training                      = training,
    # evaluation
    evaluation                    = evaluation,
    # optimizer
    lr_scheduler                  = lr_scheduler,
)
