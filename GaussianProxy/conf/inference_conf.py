from dataclasses import dataclass
from pathlib import Path

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torch import dtype

from GaussianProxy.conf.training_conf import DataSet, EvaluationStrategy


@dataclass(kw_only=True)
class ProfileConfig:
    enabled: bool = False
    record_shapes: bool = False
    profile_memory: bool = True
    with_stack: bool = True
    with_flops: bool = False
    export_chrome_trace: bool = False


@dataclass(kw_only=True)
class InferenceConfig:
    # Choose the experiment (trained model weights)
    root_experiments_path: Path
    project_name: str
    run_name: str
    saved_model_foldername: str = "best_model"

    # Scheduler config
    scheduler_config_path: Path | str | None = None
    scheduler_type: type[SchedulerMixin] | None = None
    import_orig_config: bool = True

    # Output directory (where to put the generated images / tensors)
    output_dir: Path

    # Device (ignored if launched with accelerate)
    device: str

    # Optimizations
    compile: bool
    dtype: dtype

    # Data
    dataset: DataSet
    true_label_time_0: str | None = None

    # Evaluations
    evaluation_strategies: list[EvaluationStrategy]

    # Profiling
    profiling: ProfileConfig

    # Debug
    debug: bool = False

    # Temp Dir
    tmpdir_location: str
