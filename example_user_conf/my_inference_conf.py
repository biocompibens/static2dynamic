# ruff: noqa: F401

from pathlib import Path

from torch import float32

from GaussianProxy.conf.inference_conf import InferenceConfig, ProfileConfig
from GaussianProxy.conf.training_conf import (
    EvaluationStrategy,
    ForwardNoising,
    ForwardNoisingLinearScaling,
    InversionRegenerationOnly,
    InvertedRegeneration,
    IterativeInvertedRegeneration,
    MetricsComputation,
    SimilarityWithTrainData,
    SimpleGeneration,
    VideoGenerationFromNoise,
)

# isort: off
# -------------------------------------------- Dataset --------------------------------------------
from my_conf.dataset.dataset_conf import dataset

# --------------------------------------------- Model ---------------------------------------------
root_experiments_path = Path("path", "to", "root", "experiment", "paths")
project_name = "project_name"  # typically the training run
folder_name = "folder_name_for_this_inference_run"

run_path = root_experiments_path / project_name / folder_name
assert run_path.exists(), f"'{run_path}' does not exist"

# ------------------------------------------- Scheduler -------------------------------------------
scheduler_type = None  # defaults to DDIM

# ------------------------------------------ Evaluations ------------------------------------------
# fmt: off
eval_strats = [
    InvertedRegeneration(
        nb_inversion_diffusion_timesteps = 300,
        nb_diffusion_timesteps           = 300,
        nb_video_timesteps               = 100,
        nb_video_times_in_parallel       = 6,
        nb_generated_samples             = 16,
        n_rows_displayed                 = 4,
        plate_name_to_simulate           = None,
        gen_times_type                   = "evenly_spaced_from_inversion",
        split                            = "test",
    )
]
# fmt: on

# ------------------------------------------ Final Config -----------------------------------------
# fmt: off
inference_conf = InferenceConfig(
    # Choose the experiment (== trained model weights)
    root_experiments_path       = root_experiments_path,
    project_name                = project_name,
    run_name                    = folder_name,
    saved_model_foldername      = "best_model",
    # Choose a custom scheduler
    scheduler_type              = scheduler_type,
    scheduler_config_path       = None,
    import_orig_config          = True,
    # Output directory (where to put the generated images / tensors)
    output_dir                  = run_path / "inferences",
    # Device
    device                      = "cuda:1",
    # Optimizations
    compile                     = True,
    dtype                       = float32,
    # Data
    dataset                     = dataset,
    true_label_time_0           = None,
    # Evaluations
    evaluation_strategies       = eval_strats, # type: ignore[reportArgumentType]
    # Profiling
    profiling                   = ProfileConfig(), # off by default
    # Debug
    debug                       = False,
    # Temp Dir
    tmpdir_location             = "/tmp",
)
