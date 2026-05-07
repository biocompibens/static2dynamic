import functools
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import MISSING


def custom_showwarning(message, category, filename, lineno, _file=None, _line=None):
    print(f"{filename}:{lineno}: {category.__name__}: {message}")


warnings.showwarning = custom_showwarning
warnings.simplefilter("once")


@dataclass(kw_only=True)
class Slurm:
    enabled: bool
    monitor: bool = False
    total_job_time: int | None = None  # in minutes
    send_timeout_signal_n_minutes_before_end: int = 5  # in minutes
    email: str
    output_folder: str
    num_gpus: int
    qos: str
    constraint: str | None
    nodes: int
    account: str
    max_num_requeue: int
    partition: str | None
    job_launch_delay_hours: int | None = None  # in hours

    def __post_init__(self):
        valid_qos_values = ["dev", "t3", "t4"]
        if self.qos not in valid_qos_values:
            raise ValueError(f"Invalid qos value. Expected one of {valid_qos_values}, got {self.qos}")


@dataclass
class AccelerateLaunchArgs:
    machine_rank: int
    num_machines: int
    rdzv_backend: str
    same_network: str
    mixed_precision: str
    num_processes: int | None
    main_process_port: int
    dynamo_backend: str | None = "no"
    dynamo_plugin: dict[str, Any] | None = None
    gpu_ids: str = "all"
    multi_gpu: bool = field(init=False)

    def __post_init__(self):
        cond1 = self.num_processes is not None and self.num_processes > 1
        cond2 = len([gpu_id for gpu_id in self.gpu_ids.split(",") if gpu_id]) > 1
        if cond1 or cond2:
            self.multi_gpu = True
        else:
            self.multi_gpu = False


@dataclass
class Accelerate:
    launch_args: AccelerateLaunchArgs
    offline: bool  # TODO: move this arg that does not belong here (make it general like debug)


@dataclass
class DatasetParams:  # TODO: fusion with DataSet
    """
    - `file_extension`: the extension of the files to load, without the dot
    - `key_transform`: a function to transform the subdir name into a timestep
    - `sorting_func`: a function to sort the subdirs
    - `dataset_class`: the class of the dataset to instantiate
    """

    file_extension: str
    key_transform: Any  # should be Callable[[str], int] | Callable[[str], str] ...
    sorting_func: Any  # should be Callable[something...]
    dataset_class: type


@dataclass(kw_only=True)
class DataSet:
    # data_shape should be tuple[int, int, int] | tuple[int, int], but unions of containers
    # are not yet supported by OmegaConf: https://github.com/omry/omegaconf/issues/144
    data_shape: tuple[int, ...]
    path: str | Path = field(default=MISSING)
    transforms: Any
    name: str
    expected_initial_data_range: tuple[float, float] | None
    # same goes for selected_dists: should be list[int] | list[str]...
    selected_dists: list | None = None
    dataset_params: DatasetParams | None = None
    fully_ordered: bool = False
    path_to_single_parquet: str | None = None
    path_to_train_test_labels_parquet: str | None = None
    hard_aug_path: Path | None = None
    separate_gt_starting_class_path: str | None = None

    def __post_init__(self):
        """Checks"""
        # Check fully order config consistency
        import inspect

        if self.dataset_params is not None:
            if inspect.isclass(self.dataset_params.dataset_class):
                ds_class = self.dataset_params.dataset_class
            elif isinstance(self.dataset_params.dataset_class, functools.partial):
                ds_class = self.dataset_params.dataset_class.func
            else:
                raise TypeError(
                    f"dataset_params.dataset_class should be a class or a functools.partial, got {type(self.dataset_params.dataset_class)}"
                )
        else:
            ds_class = None

        from GaussianProxy.utils.data import BaseContinuousTimeDataset  # avoid circular import

        fully_ordered_conditions = (
            self.fully_ordered,
            ds_class is not None and issubclass(ds_class, BaseContinuousTimeDataset),
        )
        if any(fully_ordered_conditions):
            err_msg = f"Dataset {self.name} is marked as fully ordered, but not all conditions are correctly set: fully_ordered={self.fully_ordered}"
            if self.dataset_params is not None:
                err_msg += f", dataset_params.dataset_class={self.dataset_params.dataset_class}"
            if not all(fully_ordered_conditions):
                raise ValueError(err_msg)

        if self.path_to_single_parquet is not None and not self.fully_ordered:
            raise ValueError(f"Dataset {self.name} has path_to_single_parquet set but is not marked as fully ordered.")

        if self.path_to_train_test_labels_parquet is not None and not self.fully_ordered:
            raise ValueError(
                f"Dataset {self.name} has path_to_train_test_labels_parquet set but is not marked as fully ordered."
            )


@dataclass
class DataLoader:
    num_workers: int | None
    train_prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool


@dataclass(kw_only=True)
class Training:
    gradient_accumulation_steps: int
    train_batch_size: int
    max_grad_norm: int
    nb_time_samplings: int
    unpaired_data: bool
    as_many_samples_as_unpaired: bool = False
    reweight_sampling: bool


@dataclass(kw_only=True)
class EvaluationStrategy:
    nb_diffusion_timesteps: int
    name: str = field(default=MISSING)
    # train script requires a name that starts with the class name!!!


@dataclass(kw_only=True)
class InversionRegenerationOnly(EvaluationStrategy):
    nb_generated_samples: int
    plate_name_to_simulate: str | None = None
    n_rows_displayed: int

    def __post_init__(self):
        # if not set at init, set name based on parameters
        if self.name == MISSING:
            self.name = f"InversionRegenerationOnly_{self.nb_diffusion_timesteps}diffsteps"


@dataclass(kw_only=True)
class InvertedRegeneration(EvaluationStrategy):
    """
    - `gen_times_type`: One of:
        - `"evenly_spaced"`
        - `"evenly_spaced_from_inversion"`
        - `"evenly_spaced_from_inversion_reversed"`
        - `list[float]`: a list of times at which to generate samples
        - `tuple[str, float] | tuple[str, tuple[float, float]]`: a 2-tuple with the first element being one of the previous strings, and the second element being a float or a 2-tuple of floats indicating the start/end times for the evenly spaced times generation

    - `split`: `"train"`, `"test"`, or `None` (= all data)
    - `also_save_individual_videos_separately`: Whether to save individual videos separately, per frame
    """

    nb_generated_samples: int
    plate_name_to_simulate: str | None = None
    nb_video_times_in_parallel: int
    nb_video_timesteps: int
    n_rows_displayed: int
    nb_inversion_diffusion_timesteps: int | None = None
    selected_times: str | None | Path = None
    gen_times_type: Any  # should be: str | list[float] | tuple[str, float | int | tuple[float | int, float | int]]
    split: str | None = "test"
    also_save_individual_videos_separately: bool = False

    def __post_init__(self):
        # if not set at init, set name based on parameters
        if self.name == MISSING or self.name == "":
            name = f"InvertedRegeneration_{self.nb_diffusion_timesteps}diffsteps_{self.nb_video_timesteps}vidsteps"
            if self.nb_inversion_diffusion_timesteps is not None:
                name += f"_{self.nb_inversion_diffusion_timesteps}invsteps"
            if self.plate_name_to_simulate is not None:
                name += f"_{self.plate_name_to_simulate}"
            self.name = name
            if self.selected_times is not None:
                self.name += "_selected_times"


@dataclass(kw_only=True)
class IterativeInvertedRegeneration(InvertedRegeneration):
    def __post_init__(self):
        # if not set at init, set name based on parameters
        if self.name == MISSING:
            name = f"IterativeInvertedRegeneration_{self.nb_diffusion_timesteps}diffsteps_{self.nb_video_timesteps}vidsteps"
            if self.nb_inversion_diffusion_timesteps is not None:
                name += f"_{self.nb_inversion_diffusion_timesteps}invsteps"
            if self.plate_name_to_simulate is not None:
                name += f"_{self.plate_name_to_simulate}"
            self.name = name


@dataclass(kw_only=True)
class SimpleGeneration(EvaluationStrategy):
    n_rows_displayed: int
    nb_generated_samples: int

    def __post_init__(self):
        # if not set at init, set name based on parameters
        if self.name == MISSING:
            self.name = f"SimpleGeneration_{self.nb_diffusion_timesteps}diffsteps"


@dataclass(kw_only=True)
class VideoGenerationFromNoise(EvaluationStrategy):
    n_rows_displayed: int
    nb_generated_samples: int
    nb_video_frames: int
    nb_video_frames_in_parallel: int

    def __post_init__(self):
        # if not set at init, set name based on parameters
        if self.name == MISSING:
            self.name = (
                f"VideoGenerationFromNoise_{self.nb_diffusion_timesteps}diffsteps_{self.nb_video_frames}vidsteps"
            )


@dataclass(kw_only=True)
class SimilarityWithTrainData(EvaluationStrategy):
    nb_generated_samples: int
    batch_size: int
    nb_batches_shown: int
    metrics: Any = "cosine"  # should be list[Literal["cosine", "L2"]]
    n_rows_displayed: int

    def __post_init__(self):
        # if not set at init, set name based on parameters
        if self.name == MISSING:
            self.name = f"SimilarityWithTrainData_{self.metrics}"


@dataclass(kw_only=True)
class ForwardNoising(EvaluationStrategy):
    forward_noising_frac: float
    nb_generated_samples: int
    plate_name_to_simulate: str | None = None
    nb_video_times_in_parallel: int
    nb_video_timesteps: int
    n_rows_displayed: int

    def __post_init__(self):
        # set name based on parameters
        if self.name == MISSING:
            name = f"ForwardNoising_{self.nb_diffusion_timesteps}diffsteps_{self.nb_video_timesteps}vidsteps"
            if self.forward_noising_frac is not None:
                name += f"_{self.forward_noising_frac}frac"
            if self.plate_name_to_simulate is not None:
                name += f"_{self.plate_name_to_simulate}"
            self.name = name


@dataclass(kw_only=True)
class ForwardNoisingLinearScaling(ForwardNoising):
    forward_noising_frac_start: float
    forward_noising_frac_end: float

    def __post_init__(self):
        # if not set at init, set name based on parameters
        if self.name == MISSING:
            self.name = "ForwardNoisingLinearScaling_"
            self.name += f"{self.nb_diffusion_timesteps}diffsteps_"
            self.name += f"{self.nb_video_timesteps}vidsteps"
            self.name += f"_{self.forward_noising_frac_start}start_{self.forward_noising_frac_end}end"


@dataclass(kw_only=True)
class MetricsComputation(EvaluationStrategy):
    """
    - feature_extractor: 'inception-v3-compat', 'dinov2-vit-{s,b,l,g}-14', etc. see https://torch-fidelity.readthedocs.io/en/latest/registry.html#preregistered-feature-extractors
    - nb_samples_to_gen_per_time: int or 'adapt'+{'half','aug'} or (int, 'aug')
    - metrics_comparison_mode: one of
        - 'gen_vs_true_same_time'
        - 'gen_vs_true_all_other_times'
        - 'true_vs_true_same_time'
        - 'true_vs_true_all_other_times'
    - max_nb_reference_samples: if set, randomly subsample each true reference dataset down to that many samples
    """

    nb_samples_to_gen_per_time: Any  # should be int | str | tuple[int, str]...
    batch_size: int
    regen_images: bool = True
    selected_times: list
    dtype: str = "float32"
    augmentations_for_metrics_comp: list[str]
    also_compute_metrics_on_all_times: bool = False
    metrics_comparison_mode: str = "gen_vs_true_same_time"
    max_nb_reference_samples: int | None = None
    feature_extractor: str = "inception-v3-compat"

    def __post_init__(self):
        valid_modes = {
            "gen_vs_true_same_time",
            "gen_vs_true_all_other_times",
            "true_vs_true_same_time",
            "true_vs_true_all_other_times",
        }
        if self.metrics_comparison_mode not in valid_modes:
            raise ValueError(
                f"Invalid metrics_comparison_mode: {self.metrics_comparison_mode}. Expected one of {sorted(valid_modes)}"
            )

        # if not set at init, set name based on parameters
        nb_samples_str = str(self.nb_samples_to_gen_per_time).replace(" ", "_").translate(str.maketrans("", "", "()',"))
        if self.name == MISSING:
            self.name = f"MetricsComputation_{nb_samples_str}_samples"


@dataclass
class Evaluation:
    every_n_opt_steps: int | None
    batch_size: int
    nb_video_timesteps: int
    strategies: list[EvaluationStrategy]


@dataclass(kw_only=True)
class Checkpointing:
    checkpoints_total_limit: int
    resume_from_checkpoint: bool | int | str
    checkpoint_every_n_steps: int
    chckpt_base_path: Path


@dataclass(kw_only=True)
class DDIMSchedulerConfig:
    num_train_timesteps: int
    prediction_type: str  # 'epsilon' or 'v_prediction'
    # clipping
    clip_sample: bool
    clip_sample_range: float
    # dynamic thresholding
    thresholding: bool
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1
    # noise scheduler
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    timestep_spacing: str
    rescale_betas_zero_snr: bool


@dataclass
class UNet2DModelConfig:
    sample_size: int
    in_channels: int
    out_channels: int
    down_block_types: tuple[str, ...]
    up_block_types: tuple[str, ...]
    block_out_channels: tuple[int, ...]
    layers_per_block: int
    act_fn: str
    class_embed_type: str | None
    center_input_sample: bool = False
    time_embedding_type: str = "positional"
    freq_shift: int = 0
    flip_sin_to_cos: bool = True
    mid_block_scale_factor: float = 1
    downsample_padding: int = 1
    dropout: float = 0.0
    attention_head_dim: int = 8
    norm_num_groups: int = 32
    norm_eps: float = 1e-5
    resnet_time_scale_shift: str = "default"


@dataclass(kw_only=True)
class UNet2DConditionModelConfig:
    sample_size: int
    in_channels: int
    out_channels: int
    down_block_types: tuple[str, ...]
    up_block_types: tuple[str, ...]
    block_out_channels: tuple[int, ...]
    norm_num_groups: int = 32
    layers_per_block: int
    act_fn: str
    cross_attention_dim: int


@dataclass
class TimeEncoderConfig:
    encoding_dim: int
    time_embed_dim: int
    flip_sin_to_cos: bool
    downscale_freq_shift: float


@dataclass(kw_only=True)
class OneCycleLRConfig:
    name: str = "OneCycleLRConfig"
    max_lr: float
    pct_start: float = 0.3
    anneal_strategy: str = "cos"  # 'cos' or 'linear'
    div_factor: float = 10
    final_div_factor: float = 1e2


@dataclass(kw_only=True)
class LinearLRConfig:
    name: str = "LinearLRConfig"
    start_factor: float = 1 / 3
    end_factor: float = 1
    base_lr: float


LRConfig = OneCycleLRConfig | LinearLRConfig


@dataclass(kw_only=True)
class Config:
    # Defaults
    defaults: list

    # Model
    dynamic: DDIMSchedulerConfig
    net: Any  # Unions of containers are not supported.......
    time_encoder: TimeEncoderConfig

    # Script
    launcher_script_parent_folder: str
    script: str

    # Experiment Variables
    exp_parent_folder: str
    project: str
    run_name: str

    # Hydra
    hydra: Any

    # Slurm
    slurm: Slurm

    # Accelerate
    accelerate: Accelerate

    # Miscellaneous
    debug: bool
    profile: bool = False
    resume_method: str = "rewind"  # Literal["fork", "rewind", "new_run", "resume"]
    diff_mode: str = "config_only"  # "config_only" or "full"

    # Caches
    tmpdir_location: str | None = None

    # Experiment tracker
    logger: str
    entity: str

    # Checkpointing
    checkpointing: Checkpointing

    # Dataset
    dataset: DataSet

    # Dataloader
    dataloaders: DataLoader

    # Training
    training: Training

    # Evaluation
    evaluation: Evaluation

    # Optimization
    lr_scheduler: Any  # should be LRConfig...

    def __post_init__(self):
        """Checks"""
        # dataset
        if not isinstance(self.dataset, DataSet):
            warnings.warn(
                "Cannot check dataset config because it is not instantiated yet",
                RuntimeWarning,
            )
        else:
            for eval_strat in self.evaluation.strategies:
                if isinstance(eval_strat, MetricsComputation):
                    if self.dataset.selected_dists is not None:
                        if not set(eval_strat.selected_times).issubset(set(self.dataset.selected_dists)):
                            raise ValueError(
                                f"MetricsComputation selected_times {eval_strat.selected_times} not in dataset"
                            )
                    else:
                        warnings.warn(
                            f"Cannot check if MetricsComputation's selected_times {eval_strat.selected_times} are in dataset {self.dataset.name} because no selected_dists were provided",
                            RuntimeWarning,
                        )
        # evaluations
        if any(isinstance(eval_strat, SimilarityWithTrainData) for eval_strat in self.evaluation.strategies):
            sim_strat_index = next(
                index
                for index, strategy in enumerate(self.evaluation.strategies)
                if isinstance(strategy, SimilarityWithTrainData)
            )
            try:
                metrics_comp_index = next(
                    index
                    for index, strategy in enumerate(self.evaluation.strategies)
                    if isinstance(strategy, MetricsComputation)
                )
            except StopIteration as e:
                raise ValueError(
                    f"SimilarityWithTrainData strategy requires MetricsComputation strategy to be present in evaluation strategies, got: {[s.name for s in self.evaluation.strategies]}"
                ) from e
            if sim_strat_index < metrics_comp_index:
                raise ValueError(
                    "SimilarityWithTrainData strategy must come after MetricsComputation strategy in evaluation strategies"
                )

        if any(
            isinstance(eval_strat, MetricsComputation) for eval_strat in self.evaluation.strategies
        ) and self.net.out_channels not in (3, 1):
            raise ValueError(
                f"MetricsComputation only supports RGB (and apparently gray-level??) images, got net.out_channels={self.net.out_channels}"
            )

        if (
            self.accelerate.launch_args.dynamo_plugin is not None
            and self.accelerate.launch_args.dynamo_backend is not None
        ):
            raise ValueError("You cannot pass in both `dynamo_plugin` and `dynamo_backend`, please only pass in one.")
