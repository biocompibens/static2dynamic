print("Loading imports in inference.py...", flush=True, end="")
# ruff: noqa:E402
import time as system_time

start_import_time = system_time.time()
import ast
import json
import logging
import operator
import os
import pickle
import random
import re
import shutil
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from math import ceil
from pathlib import Path
from typing import Literal

import colorlog
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch_fidelity
from accelerate import Accelerator, DistributedType
from accelerate.logging import MultiProcessAdapter
from accelerate.utils import InitProcessGroupKwargs
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from enlighten import Manager, get_manager
from numpy import ndarray
from PIL import Image, ImageDraw
from torch import Tensor
from torch.nn import CosineSimilarity, PairwiseDistance
from torch.profiler import (
    ProfilerActivity,
    profile,
)  # , tensorboard_trace_handler #TODO:gh-pytorch#136040
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Normalize as Normalize_v1
from torchvision.transforms.v2 import Compose, ToDtype
from torchvision.transforms.v2 import Normalize as Normalize_v2
from torchvision.utils import make_grid

from GaussianProxy.conf.inference_conf import InferenceConfig
from GaussianProxy.conf.training_conf import (
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
from GaussianProxy.utils.data import (
    BaseContinuousTimeDataset,
    BaseDataset,
    ContinuousTimeImageDataset,
    ContinuousTimeImageDataset1D,
    ImageDataset,
    ImageDataset1Dto3D,
    remove_flips_and_rotations_from_transforms,
)
from GaussianProxy.utils.misc import (
    ACCEPTED_NORMALIZATIONS,
    bold,
    generate_all_augs,
    get_evenly_spaced_timesteps,
    hard_augment_dataset_all_square_symmetries,
    normalize_elements_for_logging,
    save_images_for_metrics_compute,
    shorten,
    warn_about_dtype_conv,
)
from GaussianProxy.utils.models import VideoTimeEncoding

end_import_time = system_time.time()
print(f" done! (took {end_import_time - start_import_time:.1f} seconds)", flush=True)

# No grads
torch.set_grad_enabled(False)

# Speed up
torch.backends.fp32_precision = "tf32"  # pyright: ignore[reportAttributeAccessIssue]
# also set the old flags, otherwise inductor fails
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# do benchmarking
torch.backends.cudnn.benchmark = True


class _ShortNameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Ensure a short_name attribute is available for terminal formatter
        try:
            record.short_name = shorten(record.name)
        except Exception:
            record.short_name = record.name
        return True


def get_distrib_logger(inference_conf: InferenceConfig) -> MultiProcessAdapter:
    """Handles distributed logging"""
    term_handler = logging.StreamHandler(sys.stdout)
    term_handler.setFormatter(
        colorlog.ColoredFormatter(
            "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(short_name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",  # no milliseconds in terminal
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
    )
    term_handler.addFilter(_ShortNameFilter())
    term_handler.setLevel(logging.DEBUG if inference_conf.debug else logging.INFO)

    log_file_path = inference_conf.output_dir / "logs.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    file_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(Path(__file__).stem)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(term_handler)
    logger.addHandler(file_handler)
    return MultiProcessAdapter(logger, {})


PADDING = 2  # padding between images in the grid; ffmpeg warns if end image is not divisible by 16


class SchedulersCommonClass(SchedulerMixin, ConfigMixin):
    pass


@torch.inference_mode()
def main(cfg: InferenceConfig, logger: MultiProcessAdapter, config_snapshot: str, config_filename: str) -> None:
    ###############################################################################################
    #                                          Accelerator
    ###############################################################################################
    accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))  # 2 hours before NCCL timeout
    accelerator = Accelerator(
        kwargs_handlers=[accelerator_kwargs],
    )

    ###############################################################################################
    #                                            Logger
    ###############################################################################################
    logger.info("#\n" + "#" * 120 + "\n#" + " " * 47 + "Starting inference script" + " " * 46 + "#\n" + "#" * 120)

    ###############################################################################################
    #                                          Check paths
    ###############################################################################################
    # where to load the model/schedulers from
    project_path = cfg.root_experiments_path / cfg.project_name
    assert project_path.exists(), f"Project path {project_path} does not exist."
    run_path = project_path / cfg.run_name
    assert run_path.exists(), f"Run path {run_path} does not exist."
    logger.info(f"run path: {run_path}")
    assert (run_path / cfg.saved_model_foldername).exists(), (
        f"Saved model folder '{cfg.saved_model_foldername}' does not exist."
    )

    # where to write the outputs to
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"output dir: {cfg.output_dir}")

    ###############################################################################################
    #                                          Load Model
    ###############################################################################################
    if accelerator.distributed_type == DistributedType.NO:
        device = cfg.device
        logger.info(f"Using device {device}")
    else:
        device = accelerator.device

    # denoiser
    net: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(  # pyright: ignore[reportAssignmentType]
        run_path / cfg.saved_model_foldername / "net", local_files_only=True
    )
    nb_params_M = round(net.num_parameters() / 1e6)
    logger.info(f"Loaded denoiser from {cfg.saved_model_foldername}/net with ~{nb_params_M}M parameters")
    warn_about_dtype_conv(net, cfg.dtype, logger)
    net.to(device, cfg.dtype)  # pyright: ignore[reportArgumentType]
    if cfg.compile:
        net = torch.compile(net)  # pyright: ignore[reportAssignmentType]

    # time encoder
    video_time_encoder: VideoTimeEncoding = VideoTimeEncoding.from_pretrained(  # pyright: ignore[reportAssignmentType]
        run_path / cfg.saved_model_foldername / "video_time_encoder", local_files_only=True
    )
    nb_params_K = round(video_time_encoder.num_parameters() / 1e3)
    logger.info(
        f"Loaded video time encoder from {cfg.saved_model_foldername}/video_time_encoder with ~{nb_params_K}K parameters"
    )
    warn_about_dtype_conv(video_time_encoder, cfg.dtype, logger)
    video_time_encoder.to(device, cfg.dtype)  # pyright: ignore[reportArgumentType]

    # dynamic
    orig_dynamic: DDIMScheduler = DDIMScheduler.from_pretrained(
        run_path / cfg.saved_model_foldername / "dynamic", local_files_only=True
    )
    logger.debug(f"Loaded original dynamic from {run_path / cfg.saved_model_foldername / 'dynamic'}:\n{orig_dynamic}")

    dynamic_type = DDIMScheduler
    dynamic: SchedulersCommonClass

    if cfg.scheduler_type is not None:
        logger.info(f"Using a {cfg.scheduler_type} scheduler")
        dynamic_type = cfg.scheduler_type

    if cfg.scheduler_config_path is not None:
        logger.info(f"Loading scheduler config from {cfg.scheduler_config_path}")
        dynamic = dynamic_type.from_pretrained(cfg.scheduler_config_path, local_files_only=True)  # pyright: ignore[reportAssignmentType]
    elif cfg.import_orig_config:
        logger.info("Loading scheduler config from the original dynamic")
        dynamic = dynamic_type.from_config(orig_dynamic.config)  # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
    else:
        logger.info(f"Using the default {dynamic_type} config")
        dynamic = dynamic_type()  # pyright: ignore[reportAssignmentType]

    # show the diff between the two configs
    all_attrs = set(orig_dynamic.config.keys()) | set(dynamic.config.keys())
    msg = ""
    for attr in all_attrs:
        orig_val = getattr(orig_dynamic.config, attr, None)
        chosen_val = getattr(dynamic.config, attr, None)
        if orig_val != chosen_val and not attr.startswith("_"):
            msg += f"'{attr}': {orig_val} -> {chosen_val}\n"
    if len(msg) != 0:
        logger.info("Diff between original -and> chosen dynamic:\n" + msg)
    else:
        logger.info("No config difference between original and loaded dynamic")

    logger.info(f"Using dynamic:\n{dynamic}")

    ###############################################################################################
    #                               Miscellaneous common needed things
    ###############################################################################################
    assert cfg.dataset.dataset_params is not None

    database_path = Path(cfg.dataset.path)
    logger.info(f"Using dataset {cfg.dataset.name} from {database_path}")
    subdirs: list[Path] = [e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")]
    if len(subdirs) == 0:
        logger.warning(f"No subdirs found in {database_path}; using base path directly")
        subdirs = [database_path]
    else:
        logger.info(f"Found {len(subdirs)} subdirectories in {database_path}: {[s.name for s in subdirs]}")

    # use only selected_dists
    if cfg.dataset.selected_dists is not None:
        sel_subdirs_as_str = [str(i) for i in cfg.dataset.selected_dists]
        subdirs = [s for s in subdirs if s.name in sel_subdirs_as_str]
        logger.warning(f"Filtered subdirs by selected_dists {cfg.dataset.selected_dists}: {[s.name for s in subdirs]}")
        if len(subdirs) == 0:
            raise ValueError(
                f"After filtering with selected_dists {cfg.dataset.selected_dists}, no subdir was left. Available subdirs: {[s.name for s in database_path.iterdir() if s.is_dir() and not s.name.startswith('.')]}"
            )

    subdirs.sort(key=cfg.dataset.dataset_params.sorting_func)
    logger.info(f"Sorted subdirs: {[s.name for s in subdirs]}")

    ### Evenly-spaced timesteps are assumed to be the actual empirical timesteps; must change this if it changes in training code
    # TODO: actually save the timesteps used in training (well, if discrete...)
    empirical_timesteps = get_evenly_spaced_timesteps(len(subdirs))
    logger.info(f"Empirical timesteps: {[round(ts, 3) for ts in empirical_timesteps]} from {len(subdirs)} subdirs")
    timesteps2classnames: dict[float, str] = dict(zip(empirical_timesteps, [s.name for s in subdirs], strict=True))
    if cfg.true_label_time_0 is None:
        true_label_time_0 = subdirs[0].name
        logger.info(f"No true_label_time_0 specified; using the first subdir: {true_label_time_0}")
    else:
        true_label_time_0 = cfg.true_label_time_0
        logger.info(f"Using specified true_label_time_0: {true_label_time_0}")

    ###############################################################################################
    #                                       Inference passes
    ###############################################################################################
    pbar_manager: Manager = get_manager()  # pyright: ignore[reportAssignmentType]

    eval_strats_msg = "\n".join([str(eval_strat) for eval_strat in cfg.evaluation_strategies])
    logger.info(f"Running {len(cfg.evaluation_strategies)} evaluation strategies:\n{eval_strats_msg}")

    for eval_strat_idx, eval_strat in enumerate(cfg.evaluation_strategies):
        logger.info(f"Running evaluation strategy {eval_strat_idx + 1}/{len(cfg.evaluation_strategies)}:\n{eval_strat}")

        # set logger name/additional output
        logger.logger.name = eval_strat.name
        # add strategy-specific logger handler
        log_file_path = cfg.output_dir / eval_strat.name / "logs.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
        file_handler.setLevel(logging.DEBUG)
        logger.logger.addHandler(file_handler)

        # check for correct distributed environment
        is_distributed_strategy = isinstance(eval_strat, MetricsComputation)
        if not is_distributed_strategy and accelerator.distributed_type != DistributedType.NO:
            raise RuntimeError("Only MetricsComputation is supported in distributed mode")

        # run the evaluation strategy
        if type(eval_strat) is SimpleGeneration:
            simple_gen(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
                true_label_time_0,
                config_snapshot,
                config_filename,
            )
        elif type(eval_strat) is ForwardNoising:
            forward_noising(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
                true_label_time_0,
                config_snapshot,
                config_filename,
            )
        elif type(eval_strat) is ForwardNoisingLinearScaling:
            forward_noising_linear_scaling(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
                true_label_time_0,
                config_snapshot,
                config_filename,
            )
        elif type(eval_strat) is InvertedRegeneration:
            inverted_regeneration(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
                true_label_time_0,
                config_snapshot,
                config_filename,
            )
        elif type(eval_strat) is IterativeInvertedRegeneration:
            iterative_inverted_regeneration(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
                true_label_time_0,
                config_snapshot,
                config_filename,
            )
        elif type(eval_strat) is SimilarityWithTrainData:
            similarity_with_train_data(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                subdirs,
                logger,
                cfg.dataset.data_shape,  # pyright: ignore[reportArgumentType]
                config_snapshot,
                config_filename,
            )
        elif type(eval_strat) is MetricsComputation:
            ### We need the full datasets/loaders for this method
            training_run_folder = cfg.output_dir.parent
            # use the original training config in <training_run_folder>/my_conf to know if the training was with unpaired data
            try:
                training_was_with_unpaired_data = get_training_boolean_value(
                    training_run_folder / "my_conf" / "my_training_conf.py",
                    "unpaired_data",
                )
            except AttributeError as e:
                logger.warning(str(e))
                logger.warning("Assuming training was with paired data")
                training_was_with_unpaired_data = False
            metrics_computation(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                empirical_timesteps,
                timesteps2classnames,
                pbar_manager,
                logger,
                accelerator,
                training_was_with_unpaired_data,
                config_snapshot,
                config_filename,
            )
        elif type(eval_strat) is InversionRegenerationOnly:
            inversion_and_regeneration_only(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
                true_label_time_0,
                config_snapshot,
                config_filename,
            )
        elif type(eval_strat) is VideoGenerationFromNoise:
            video_generation_from_noise(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
                true_label_time_0,
                config_snapshot,
                config_filename,
            )
        else:
            raise ValueError(f"Unknown evaluation strategy {eval_strat}")

    accelerator.end_training()


def get_training_boolean_value(filepath: Path, key: str):
    with open(filepath) as f:
        tree = ast.parse(f.read(), filename=filepath)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                # Check if the value is a call to Training(...)
                if (
                    isinstance(target, ast.Name)
                    and target.id == "training"
                    and isinstance(node.value, ast.Call)
                    and getattr(node.value.func, "id", None) == "Training"
                ):
                    for kw in node.value.keywords:
                        if kw.arg == key and isinstance(kw.value, ast.Constant):
                            if not isinstance(kw.value.value, bool):
                                raise ValueError(
                                    f"Expected {key} to be a boolean, got {kw.value.value} of type {type(kw.value.value)}"
                                )
                            else:
                                return kw.value.value
    # Default to False if key is not found
    raise AttributeError(f"Could not find attribute {key} in the `training=Training(...)` assignement in {filepath}")


def simple_gen(
    cfg: InferenceConfig,
    eval_strat: SimpleGeneration,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    inference_scheduler: SchedulersCommonClass,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
    true_label_time_0: str,
    config_snapshot: str,
    config_filename: str,
):
    """
    Just simple generations.
    """
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    clean_inference_strategy_folder(base_save_path, logger, config_snapshot, config_filename)

    # 0. Setup schedulers
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using scheduler:\n{inference_scheduler}")

    # 1. Get the starting batch
    shape = get_starting_batch(
        cfg,
        eval_strat,
        logger,
        cfg.dataset.path,
        cfg.device,
        cfg.dtype,
        true_label_time_0,
        base_save_path,
    )[0].shape

    # 2. Generate the time encodings
    eval_video_times = torch.rand(shape[0], device=cfg.device, dtype=cfg.dtype)
    eval_video_times = torch.sort(eval_video_times).values  # torch.sort it for better viz
    random_video_time_enc = video_time_encoder.forward(eval_video_times)

    # 3. Generate a sample
    gen_pbar = pbar_manager.counter(
        total=len(inference_scheduler.timesteps),
        position=2,
        desc="Generating samples",
        leave=False,
    )
    gen_pbar.refresh()

    image = torch.randn(shape, device=cfg.device, dtype=cfg.dtype)

    for t in inference_scheduler.timesteps:
        model_output: torch.Tensor = net.forward(
            image,
            t,
            encoder_hidden_states=random_video_time_enc.unsqueeze(1),
            return_dict=False,
        )[0]
        image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
        gen_pbar.update()

    gen_pbar.close()

    save_grid_of_images_or_videos(
        image,
        base_save_path,
        "simple_generations",
        ["image min-max", "-1_1 raw", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        2,
        logger,
    )


def video_generation_from_noise(
    cfg: InferenceConfig,
    eval_strat: VideoGenerationFromNoise,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    inference_scheduler: SchedulersCommonClass,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
    true_label_time_0: str,
    config_snapshot: str,
    config_filename: str,
):
    """
    Generate videos not from inversions but from random noise
    """
    # 0. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    clean_inference_strategy_folder(base_save_path, logger, config_snapshot, config_filename)

    # 1. Setup schedulers
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using scheduler:\n{inference_scheduler}")

    # 2. Generate random initial noise
    shape = get_starting_batch(  # this is just to get the shape
        cfg,
        eval_strat,
        logger,
        cfg.dataset.path,
        cfg.device,
        cfg.dtype,
        true_label_time_0,
        base_save_path,
    )[0].shape
    noise = torch.randn(shape, device=cfg.device, dtype=cfg.dtype)  # (nb_generated_samples, C, H, W)
    save_grid_of_images_or_videos(
        noise.to(torch.float32),
        base_save_path,
        "starting_noise",
        ["image 5perc-95perc", "image min-max"],
        eval_strat.n_rows_displayed,
        2,
        logger,
    )

    # 2. Compute time encodings
    video_times = torch.linspace(0, 1, steps=eval_strat.nb_video_frames, device=cfg.device, dtype=cfg.dtype)
    video_time_enc = video_time_encoder.forward(video_times)  # (nb_video_frames, tim_embed_dim)

    # 3. Generate videos
    # the generation is parallelized along video time, but this is useful
    # only if a small inference batch size is used
    video = []
    nb_vid_batches = ceil(eval_strat.nb_video_frames / eval_strat.nb_video_frames_in_parallel)

    video_time_pbar = pbar_manager.counter(
        total=nb_vid_batches,
        position=1,
        desc="Video frames batches",
        leave=False,
    )
    video_time_pbar.refresh()

    for vid_batch_idx in range(nb_vid_batches):
        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps" + " " * 4,
            leave=False,
        )
        diff_time_pbar.refresh()

        start = vid_batch_idx * eval_strat.nb_video_frames_in_parallel
        end = (vid_batch_idx + 1) * eval_strat.nb_video_frames_in_parallel
        this_batch_video_time_enc = video_time_enc[start:end]  # (nb_video_frames_in_parallel, tim_embed_dim)

        # duplicate the noise for each video frame
        image = torch.cat([noise.clone() for _ in range(this_batch_video_time_enc.shape[0])])
        # shape: (nb_video_frames_in_parallel * nb_generated_samples, C, H, W)

        # duplicate the video time for each sample
        this_batch_video_time_enc = torch.repeat_interleave(
            this_batch_video_time_enc, eval_strat.nb_generated_samples, dim=0
        )  # (nb_video_frames_in_parallel * nb_generated_samples, tim_embed_dim)

        for t in inference_scheduler.timesteps:
            model_output: torch.Tensor = net.forward(
                image,
                t,
                encoder_hidden_states=this_batch_video_time_enc.unsqueeze(1),
                return_dict=False,
            )[0]
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()
        video.append(image)
        video_time_pbar.update()

    video_time_pbar.close()

    # video is a list of ceil(nb_video_frames / nb_video_frames_in_parallel) elements, each of shape
    # (nb_video_frames_in_parallel * nb_generated_samples, C, H, W)
    video = torch.cat(video)  # (nb_video_frames * nb_generated_samples, C, H, W)
    video = video.split(eval_strat.nb_generated_samples)  # tuple (nb_video_frames, nb_generated_samples, C, H, W)
    video = torch.stack(video)  # (nb_video_frames, nb_generated_samples, C, H, W)
    # save_images_or_videos expects (video_time, nb_generated_samples, C, H, W)
    expected_video_shape = (
        eval_strat.nb_video_frames,
        eval_strat.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    # 2 separate calls here to only save the individual frames of the -1_1 raw version
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "videos",
        ["image min-max", "video min-max", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        2,
        logger,
    )
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "videos",
        ["-1_1 raw"],
        eval_strat.n_rows_displayed,
        2,
        logger,
        also_save_individual_frames=True,
    )


def similarity_with_train_data(
    cfg: InferenceConfig,
    eval_strat: SimilarityWithTrainData,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    inference_scheduler: SchedulersCommonClass,
    pbar_manager: Manager,
    subdirs: list[Path],
    logger: MultiProcessAdapter,
    sample_shape: tuple[int, int, int],
    config_snapshot: str,
    config_filename: str,
):
    """
    Test model memorization by:

    - generating n samples (from random noise)
    - computing the closest (generated, true) pair by similarity, for each generated image
    - plotting the distribution of these n closest similarities
    - showing the p < n closest pairs

    All computations are forcefully performed on fp32 for numerical precision.

    Similarities can be Euclidean cosine, L2, or both.
    """
    # Checks
    assert cfg.dataset.dataset_params is not None

    # -1. Prepare output directory & change models to fp16
    base_save_path = cfg.output_dir / eval_strat.name
    clean_inference_strategy_folder(base_save_path, logger, config_snapshot, config_filename)

    if cfg.dtype != torch.float32:
        logger.warning(
            "Switching to fp32 for this evaluation strategy as we need high numerical precision to avoid discretization artifacts in the histograms."
        )
        net = net.to(torch.float32)  # pyright: ignore[reportArgumentType]
        video_time_encoder = video_time_encoder.to(torch.float32)  # pyright: ignore[reportArgumentType]

    # 0. Setup schedulers
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using scheduler:\n{inference_scheduler}")

    # 1. Setup the giant dataset of all datasets
    all_samples = list(subdirs[0].parent.rglob(f"*/*.{cfg.dataset.dataset_params.file_extension}"))
    kept_transforms, removed_transforms = remove_flips_and_rotations_from_transforms(cfg.dataset.transforms)
    all_times_ds: BaseDataset = cfg.dataset.dataset_params.dataset_class(
        all_samples,
        kept_transforms,  # no augs, they will be manually performed
        cfg.dataset.expected_initial_data_range,
    )
    logger.debug(f"Built all-times dataset from {subdirs[0].parent}:\n{all_times_ds}")

    # TODO: 2. Compute the average image of the dataset (to remove it afterwards)

    # 3. Generate samples and compute closest similarities
    num_full_batches, remaining = divmod(eval_strat.nb_generated_samples, eval_strat.batch_size)
    actual_bses = [eval_strat.batch_size] * num_full_batches + ([remaining] if remaining != 0 else [])

    batches_pbar = pbar_manager.counter(
        total=eval_strat.nb_generated_samples,
        position=1,
        desc=f"Generating samples (batch size: {eval_strat.batch_size})",
        leave=False,
    )
    batches_pbar.refresh()

    # instantiate similarities
    if isinstance(eval_strat.metrics, str):
        eval_strat.metrics = [eval_strat.metrics]
    metrics: dict[str, Callable] = dict.fromkeys(eval_strat.metrics)  # pyright: ignore[reportAssignmentType]
    for metric in eval_strat.metrics:
        if metric == "cosine":
            metrics[metric] = lambda x, y: CosineSimilarity(dim=1, eps=1e-9)(x, y)
        elif metric == "L2":
            metrics[metric] = PairwiseDistance(p=2, eps=1e-9)
        else:
            raise ValueError(f"Unsupported metric {metric}; expected 'cosine' or 'L2'")

    # get augmentation factor
    augmented_imgs = generate_all_augs(torch.randn((128, 128, 3)), removed_transforms)
    aug_factor = len(augmented_imgs)
    logger.debug(f"Augmentation factor: {aug_factor}")
    all_sims = {
        metric_name: torch.full(
            (eval_strat.nb_generated_samples, len(all_times_ds), aug_factor),
            float("NaN"),
            device=cfg.device,
            dtype=torch.float32,
        )
        for metric_name in metrics
    }

    BEST_VAL = {"cosine": 0, "L2": float("inf")}
    COMPARISON_OPERATORS = {
        "cosine": operator.gt,
        "L2": operator.lt,
    }
    MAX_OR_MIN = {
        "cosine": torch.maximum,
        "L2": torch.minimum,
    }
    worst_values = {
        metric_name: torch.full(
            (eval_strat.nb_generated_samples,),
            BEST_VAL[metric_name],
            device=cfg.device,
            dtype=torch.float32,
        )
        for metric_name in metrics
    }
    # closest_ds_idx_aug_idx[metric_name][i] = (closest_ds_idx, closest_aug_idx)
    closest_ds_idx_aug_idx = {
        metric_name: torch.full(
            (eval_strat.nb_generated_samples, 2),
            -1,
            device=cfg.device,
            dtype=torch.int64,
        )
        for metric_name in metrics
    }

    for batch_idx, bs in enumerate(actual_bses):
        start = batch_idx * eval_strat.batch_size
        end = start + bs

        # generate samples
        eval_video_times = torch.rand(bs, device=cfg.device, dtype=torch.float32)
        random_video_time_enc = video_time_encoder.forward(eval_video_times).unsqueeze(1)

        gen_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Generating samples" + " " * 17,
            leave=False,
        )
        gen_pbar.refresh()

        generated_images = torch.randn((bs, *sample_shape), dtype=torch.float32, device=cfg.device)

        for t in gen_pbar(inference_scheduler.timesteps):
            model_output: torch.Tensor = net.forward(
                generated_images,
                t,
                encoder_hidden_states=random_video_time_enc,
                return_dict=False,
            )[0]
            generated_images = inference_scheduler.step(model_output, int(t), generated_images, return_dict=False)[0]
        gen_pbar.close()

        all_ds_pbar = pbar_manager.counter(
            total=len(all_times_ds),
            position=2,
            desc="Comparing to all training examples ",
            leave=False,
        )
        all_ds_pbar.refresh()

        # compute cosine similarities overtorch.full (augmented) all-times dataset and report largest
        generated_images = generated_images.to(torch.float32)

        all_times_dl = DataLoader(
            all_times_ds,
            batch_size=1,  # batch size *must* be 1 here
            num_workers=2,
            pin_memory=True,
            prefetch_factor=3,
            pin_memory_device=cfg.device,
        )

        for img_idx, img in enumerate(all_ds_pbar(iter(all_times_dl))):
            assert len(img) == 1, f"Expected batch size 1, got {len(img)}"
            img = img[0].to(cfg.device)
            augmented_imgs = generate_all_augs(img, removed_transforms)  # also take into account the augmentations!

            for aug_img_idx, aug_img in enumerate(augmented_imgs):
                tiled_aug_img = aug_img.unsqueeze(0).tile(bs, 1, 1, 1)
                for metric_name, metric in metrics.items():
                    value = metric(tiled_aug_img.flatten(1), generated_images.flatten(1))
                    # record all similarities
                    all_sims[metric_name][start:end, img_idx, aug_img_idx] = value
                    # update worst found indexes
                    condition = COMPARISON_OPERATORS[metric_name](value, worst_values[metric_name][start:end])
                    new_idxes = torch.where(
                        condition,
                        img_idx,
                        closest_ds_idx_aug_idx[metric_name][start:end, 0],
                    )
                    new_aug_idxes = torch.where(
                        condition,
                        aug_img_idx,
                        closest_ds_idx_aug_idx[metric_name][start:end, 1],
                    )
                    closest_ds_idx_aug_idx[metric_name][start:end, 0] = new_idxes
                    closest_ds_idx_aug_idx[metric_name][start:end, 1] = new_aug_idxes
                    # update worst found similarities
                    new_worst_values = MAX_OR_MIN[metric_name](worst_values[metric_name][start:end], value)
                    worst_values[metric_name][start:end] = new_worst_values

        all_ds_pbar.close()
        batches_pbar.update(bs)

        # plot closest pairs side-by-side
        if batch_idx < eval_strat.nb_batches_shown:
            for metric_name in metrics:
                closest_true_imgs_idxes = closest_ds_idx_aug_idx[metric_name][start:end, 0].tolist()
                aug_idxes = closest_ds_idx_aug_idx[metric_name][start:end, 1]
                closest_true_imgs = all_times_ds.__getitems__(closest_true_imgs_idxes)
                closest_true_imgs_aug = torch.stack(
                    [
                        generate_all_augs(closest_true_imgs[i], transforms=removed_transforms)[aug_idxes[i]]
                        for i in range(end - start)
                    ]
                )
                plot_side_by_side_comparison(
                    generated_images,
                    closest_true_imgs_aug,
                    base_save_path,
                    "generated_images",
                    "closest_true_images",
                    metric_name,
                    ["-1_1 raw"],
                    eval_strat.n_rows_displayed,
                    logger,
                )
    batches_pbar.close()

    # report the largest similarities
    for metric_name in metrics:
        logger.info(
            f"Worst found {metric_name} similarities: {[round(val, 3) for val in worst_values[metric_name].tolist()]}"
        )
        closest_true_imgs_names = [
            Path(all_times_ds.samples[idx]).name for idx in closest_ds_idx_aug_idx[metric_name][:, 0]
        ]
        logger.debug(f"Closest found images: {closest_true_imgs_names}")

    # torch.save all metrics and plot their histogram
    for metric_name in metrics:
        this_metric_all_sims = all_sims[metric_name].cpu()
        if torch.any(torch.isnan(this_metric_all_sims)):
            logger.warning("Found NaNs in {metric_name} similarities")
        torch.save(this_metric_all_sims, base_save_path / f"all_{metric_name}.pt")
        plt.figure(figsize=(10, 6))
        plt.hist(this_metric_all_sims.flatten().numpy(), bins=300)
        plt.title(
            f"nb_samples_generated × nb_train_samples × augment_factor = {eval_strat.nb_generated_samples} × {len(all_times_ds)} × {aug_factor} = {this_metric_all_sims.numel():,}"
        )
        plt.suptitle(f"Distribution of all {metric_name} similarities")
        plt.grid()
        plt.tight_layout()
        plt.savefig(base_save_path / f"worst_{metric_name}_hist.png")

    # plot the histogram of each per-generated-image worst similarity, for each metric
    for metric_name in metrics:
        plt.figure(figsize=(10, 6))
        plt.hist(worst_values[metric_name].flatten().cpu().numpy(), bins=300)
        plt.title(f"nb_samples_generated = {eval_strat.nb_generated_samples} = {worst_values[metric_name].numel():,}")
        plt.suptitle(f"Distribution of all {metric_name} *worst* similarities")
        plt.grid()
        plt.tight_layout()
        plt.savefig(base_save_path / f"all_{metric_name}_hist.png")


def forward_noising(
    cfg: InferenceConfig,
    eval_strat: ForwardNoising,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    inference_scheduler: SchedulersCommonClass,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
    true_label_time_0: str,
    config_snapshot: str,
    config_filename: str,
):
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    clean_inference_strategy_folder(base_save_path, logger, config_snapshot, config_filename)

    # 0. Setup scheduler
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using scheduler:\n{inference_scheduler}")

    # 0.5. Get the starting batch
    batch, _ = get_starting_batch(
        cfg,
        eval_strat,
        logger,
        cfg.dataset.path,
        cfg.device,
        cfg.dtype,
        true_label_time_0,
        base_save_path,
    )

    # 1. Save the to-be noised images
    save_grid_of_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
    )

    # 2. Sample Gaussian noise and noise the images until some step
    noise = torch.randn_like(batch)
    noise_timestep_idx = int((1 - eval_strat.forward_noising_frac) * len(inference_scheduler.timesteps))
    noise_timestep = inference_scheduler.timesteps[noise_timestep_idx].item()
    msg = (
        f"Adding noise until timestep {noise_timestep} (index {noise_timestep_idx}/{len(inference_scheduler.timesteps)}"
    )
    msg += f", timesteps range: ({inference_scheduler.timesteps.min().item()}, {inference_scheduler.timesteps.max().item()}))"
    logger.debug(msg)
    noise_timesteps: torch.IntTensor = torch.full(  # pyright: ignore[reportAssignmentType]
        (batch.shape[0],),
        noise_timestep,
        device=batch.device,
        dtype=torch.int64,
    )
    slightly_noised_sample = inference_scheduler.add_noise(batch, noise, noise_timesteps)
    save_grid_of_images_or_videos(
        slightly_noised_sample,
        base_save_path,
        "noised_samples",
        ["image min-max", "-1_1 raw", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
    )

    # 3. Generate the trajectory from it
    # the generation is parallelized along video time, but this
    # usefull if small inference batch size only
    video = []
    nb_vid_batches = ceil(eval_strat.nb_video_timesteps / eval_strat.nb_video_times_in_parallel)

    video_time_pbar = pbar_manager.counter(
        total=nb_vid_batches,
        position=1,
        desc="Video timesteps batches",
        leave=False,
    )
    video_time_pbar.refresh()

    video_times = torch.linspace(0, 1, eval_strat.nb_video_timesteps, device=cfg.device, dtype=cfg.dtype)

    for vid_batch_idx in range(nb_vid_batches):
        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps" + " " * 4,
            leave=False,
            count=noise_timestep_idx,
        )
        diff_time_pbar.refresh()

        start = vid_batch_idx * eval_strat.nb_video_times_in_parallel
        end = (vid_batch_idx + 1) * eval_strat.nb_video_times_in_parallel
        video_time_batch = video_times[start:end]
        logger.debug(f"Processing video times from {start} to {start + len(video_time_batch)}")
        # at this point video_time_batch is at most eval_strat.nb_video_times_in_parallel long;
        # we need to duplicate the video_time_encoding to match the actual batch size!
        video_time_enc = video_time_encoder.forward(video_time_batch)
        video_time_enc = video_time_enc.repeat_interleave(eval_strat.nb_generated_samples, dim=0)

        image = torch.cat([slightly_noised_sample.clone() for _ in range(len(video_time_batch))])
        # shape: (len(video_time_batch)*batch_size, channels, height, width),
        # torch.where len(video_time_batch) = eval_strat.nb_video_times_in_parallel for at least all but the last batch

        for t in inference_scheduler.timesteps[noise_timestep_idx:]:
            model_output: torch.Tensor = net.forward(
                image,
                t,
                encoder_hidden_states=video_time_enc.unsqueeze(1),
                return_dict=False,
            )[0]
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()
        video.append(image)
        video_time_pbar.update()

    video_time_pbar.close()

    # video is a list of ceil(nb_video_timesteps / nb_video_times_in_parallel) elements, each of shape
    # (len(video_time_batch) * batch_size, channels, hight, width)
    video = torch.cat(video)  # (nb_video_timesteps * nb_generated_samples, channels, height, width)
    video = video.split(eval_strat.nb_generated_samples)
    video = torch.stack(video)
    # save_images_or_videos expects (video_time, batch_size, channels, height, width)
    expected_video_shape = (
        eval_strat.nb_video_timesteps,
        eval_strat.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
    )


def forward_noising_linear_scaling(
    cfg: InferenceConfig,
    eval_strat: ForwardNoisingLinearScaling,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    inference_scheduler: SchedulersCommonClass,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
    true_label_time_0: str,
    config_snapshot: str,
    config_filename: str,
):
    # -2. Checks
    if eval_strat.nb_video_times_in_parallel != 1:
        logger.warning(
            f"nb_video_times_in_parallel was set to {eval_strat.nb_video_times_in_parallel}, but is ignored in this evaluation strategy"
        )

    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    clean_inference_strategy_folder(base_save_path, logger, config_snapshot, config_filename)

    # 0. Setup scheduler
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using scheduler:\n{inference_scheduler}")

    # 0.5. Get the starting batch
    batch, _ = get_starting_batch(
        cfg,
        eval_strat,
        logger,
        cfg.dataset.path,
        cfg.device,
        cfg.dtype,
        true_label_time_0,
        base_save_path,
    )

    # 1. Save the to-be noised images
    save_grid_of_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
    )

    # 2. Sample Gaussian noise
    noise = torch.randn_like(batch)

    # 2.5 Misc preparations
    video = []
    video_time_pbar = pbar_manager.counter(
        total=eval_strat.nb_video_timesteps,
        position=1,
        desc="Video timesteps batches",
        leave=False,
    )
    video_time_pbar.refresh()

    # video times between 0 and 1
    video_times = torch.linspace(0, 1, eval_strat.nb_video_timesteps, device=cfg.device, dtype=cfg.dtype)
    # linearly interpolate between the start and end forward_noising_fracs
    forward_noising_fracs = torch.linspace(
        eval_strat.forward_noising_frac_start,
        eval_strat.forward_noising_frac_end,
        eval_strat.nb_video_timesteps,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    logger.debug(f"Forward noising fracs: {list(forward_noising_fracs)}")

    # 3. Generate the trajectory time-per-time
    for vid_batch_idx in range(eval_strat.nb_video_timesteps):
        # noise until this timestep's index
        noise_timestep_idx = min(  # prevent potential OOB if forward_noising_frac_start is zero
            int((1 - forward_noising_fracs[vid_batch_idx]) * len(inference_scheduler.timesteps)),
            len(inference_scheduler.timesteps) - 1,
        )

        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps" + " " * 4,
            leave=False,
            count=noise_timestep_idx,
        )
        diff_time_pbar.refresh()

        video_time = video_times[vid_batch_idx]
        video_time_enc = video_time_encoder.forward(video_time.item(), batch.shape[0])

        # noise image with forward SDE
        noise_timestep = inference_scheduler.timesteps[noise_timestep_idx].item()
        msg = f"Adding noise until timestep {noise_timestep} (index {noise_timestep_idx}/{len(inference_scheduler.timesteps)}"
        msg += f", timesteps range: ({inference_scheduler.timesteps.min().item()}, {inference_scheduler.timesteps.max().item()}))"
        logger.debug(msg)
        noise_timesteps: torch.IntTensor = torch.full(  # pyright: ignore[reportAssignmentType]
            (batch.shape[0],),
            noise_timestep,
            device=batch.device,
            dtype=torch.int64,
        )
        image = inference_scheduler.add_noise(batch, noise, noise_timesteps)  # clone happens here
        # shape: (batch_size, channels, height, width)

        # denoise with backward SDE
        for t in inference_scheduler.timesteps[noise_timestep_idx:]:
            model_output: torch.Tensor = net.forward(
                image,
                t,
                encoder_hidden_states=video_time_enc.unsqueeze(1),
                return_dict=False,
            )[0]
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()
        video.append(image)
        video_time_pbar.update()

    video_time_pbar.close()

    # video is a list of nb_video_timesteps elements, each of shape (batch_size, channels, hight, width)
    video = torch.stack(video)  # (nb_video_timesteps, batch_size, channels, height, width)
    # save_images_or_videos expects (video_time, batch_size, channels, height, width)
    expected_video_shape = (
        eval_strat.nb_video_timesteps,
        eval_strat.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
    )


def inversion_and_regeneration_only(
    cfg: InferenceConfig,
    eval_strat: InversionRegenerationOnly,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    inference_scheduler: SchedulersCommonClass,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
    true_label_time_0: str,
    config_snapshot: str,
    config_filename: str,
):
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    clean_inference_strategy_folder(base_save_path, logger, config_snapshot, config_filename)

    # 0. Setup schedulers
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using scheduler:\n{inference_scheduler}")
    inverted_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(inference_scheduler.config)  # pyright: ignore[reportAssignmentType]
    inverted_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using inverted scheduler:\n{inverted_scheduler}")

    # 0.5. Get the starting batch
    batch, _ = get_starting_batch(
        cfg,
        eval_strat,
        logger,
        cfg.dataset.path,
        cfg.device,
        cfg.dtype,
        true_label_time_0,
        base_save_path,
    )

    # 1. Save the to-be noised images
    save_grid_of_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
    )

    # 2. Generate the inverted Gaussians
    inverted_gauss = batch.clone()
    inversion_video_time_enc = video_time_encoder.forward(0, batch.shape[0])

    diff_time_pbar = pbar_manager.counter(
        total=len(inverted_scheduler.timesteps),
        position=1,
        desc="Diffusion timesteps" + " " * 4,
        leave=False,
    )
    diff_time_pbar.refresh()

    for t in inverted_scheduler.timesteps:
        model_output = net.forward(
            inverted_gauss,
            t,
            encoder_hidden_states=inversion_video_time_enc.unsqueeze(1),
            return_dict=False,
        )[0]
        inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]
        diff_time_pbar.update()

    diff_time_pbar.close()

    save_grid_of_images_or_videos(
        inverted_gauss.to(torch.float32),  # needed for the 5% - 95% (numpy) normalization
        base_save_path,
        "inverted_gaussians",
        ["image min-max", "image 5perc-95perc"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
    )

    # 3. Regenerate the original starting images from it
    diff_time_pbar = pbar_manager.counter(
        total=len(inference_scheduler.timesteps),
        position=1,
        desc="Diffusion timesteps" + " " * 4,
        leave=False,
    )
    diff_time_pbar.refresh()

    video_time_enc = inversion_video_time_enc  # same time!
    image = inverted_gauss.clone()

    for t in inference_scheduler.timesteps:
        model_output: torch.Tensor = net.forward(
            image,
            t,
            encoder_hidden_states=video_time_enc.unsqueeze(1),
            return_dict=False,
        )[0]
        image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
        diff_time_pbar.update()

    diff_time_pbar.close()

    save_grid_of_images_or_videos(
        image,
        base_save_path,
        "regeneration",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
    )


def inverted_regeneration(
    cfg: InferenceConfig,
    eval_strat: InvertedRegeneration,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    inference_scheduler: SchedulersCommonClass,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
    true_label_time_0: str,
    config_snapshot: str,
    config_filename: str,
):
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    clean_inference_strategy_folder(base_save_path, logger, config_snapshot, config_filename)

    # 0. Setup schedulers
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using scheduler:\n{inference_scheduler}")
    if eval_strat.nb_inversion_diffusion_timesteps is None:
        eval_strat.nb_inversion_diffusion_timesteps = eval_strat.nb_diffusion_timesteps
    inverted_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(inference_scheduler.config)  # pyright: ignore[reportAssignmentType]
    inverted_scheduler.set_timesteps(eval_strat.nb_inversion_diffusion_timesteps)
    logger.debug(f"Using inverted scheduler:\n{inverted_scheduler}")

    # 0.5. Get the starting batch
    batch, starting_video_time = get_starting_batch(
        cfg,
        eval_strat,
        logger,
        cfg.dataset.path,
        cfg.device,
        cfg.dtype,
        true_label_time_0,
        base_save_path,
    )

    # 1. Save the to-be noised images
    save_grid_of_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
        texts=starting_video_time.float().numpy(force=True)
        if isinstance(starting_video_time, Tensor)
        else np.array([starting_video_time] * batch.shape[0]),
    )
    save_grid_of_images_or_videos(  # 2nd call to get the text-free version
        batch,
        base_save_path,
        "starting_samples_no_text",
        ["-1_1 raw"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
        save_tensor=False,
    )

    # 2. Generate the inverted Gaussians
    inverted_gauss = batch.clone()
    logger.debug(f"Starting video times for inversion: {starting_video_time}")
    inversion_video_time_enc = video_time_encoder.forward(
        starting_video_time,
        None if isinstance(starting_video_time, Tensor) else batch.shape[0],
    )

    diff_time_pbar = pbar_manager.counter(
        total=len(inverted_scheduler.timesteps),
        position=1,
        desc="Inversion diffusion timesteps" + " " * 4,
        leave=False,
    )
    diff_time_pbar.refresh()

    for t in inverted_scheduler.timesteps:
        model_output = net.forward(
            inverted_gauss,
            t,
            encoder_hidden_states=inversion_video_time_enc.unsqueeze(1),
            return_dict=False,
        )[0]
        inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]
        diff_time_pbar.update()

    diff_time_pbar.close()

    save_grid_of_images_or_videos(
        inverted_gauss.to(torch.float32),  # needed for the 5% - 95% (numpy) normalization
        base_save_path,
        "inverted_gaussians",
        ["image min-max", "image 5perc-95perc"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
        texts=starting_video_time.float().numpy(force=True).squeeze()
        if isinstance(starting_video_time, Tensor)
        else np.array([starting_video_time] * batch.shape[0]),
    )

    # 3. Generate the trajectory from it
    # the generation is parallelized along video time, but this is useful
    # only if a small inference batch size is used
    if isinstance(eval_strat.gen_times_type, tuple):
        assert len(eval_strat.gen_times_type) == 2, (
            f"Expected gen_times_type tuple of length 2, got {eval_strat.gen_times_type}"
        )
        gen_times_type = eval_strat.gen_times_type[0]
        gen_times_param = eval_strat.gen_times_type[1]
        assert isinstance(gen_times_type, str) and isinstance(gen_times_param, (float, int, tuple)), (
            f"Expected gen_times_type to be a (str, float|int|tuple), got {eval_strat.gen_times_type}"
        )
    else:
        gen_times_type = eval_strat.gen_times_type
        gen_times_param = None
    match gen_times_type:
        case "evenly_spaced":
            if gen_times_param is not None:
                assert isinstance(gen_times_param, tuple) and len(gen_times_param) == 2, (
                    f"Expected gen_times_param to be a 2-tuple, got {gen_times_param}"
                )
                start, end = gen_times_param
                assert isinstance(start, (float, int)) and isinstance(end, (float, int)), (
                    f"Expected start and end to be float or int, got {gen_times_param}"
                )
            else:
                start, end = 0, 1
            logger.warning(f"Using [{start} -> {end}] evenly spaced video times for regeneration")
            video_times = np.linspace(  # (nb_video_timesteps,) or (nb_video_timesteps, nb_generated_samples)
                start,
                end,
                eval_strat.nb_video_timesteps,
                endpoint=True,
            )
        case "evenly_spaced_from_inversion":
            if gen_times_param is not None:
                assert isinstance(gen_times_param, (float, int)), (
                    f"Expected gen_times_param to be a float or int, got {gen_times_param}"
                )
                end = gen_times_param
            else:
                end = 1
            logger.warning(f"Using [<starting time> -> {end}] evenly spaced times for regeneration")
            if isinstance(starting_video_time, Tensor) and torch.any(starting_video_time >= end).item():
                logger.error(
                    f"Some starting video times at inversion are greater than the end time {end}: {starting_video_time[starting_video_time >= end]}"
                )
            video_times = np.linspace(  # (nb_video_timesteps,) or (nb_video_timesteps, nb_generated_samples)
                starting_video_time.float().numpy(force=True)
                if isinstance(starting_video_time, Tensor)
                else starting_video_time,
                end,
                eval_strat.nb_video_timesteps,
                endpoint=True,
            )
        case "evenly_spaced_from_inversion_reversed":
            if gen_times_param is not None:
                assert isinstance(gen_times_param, (float, int)), (
                    f"Expected gen_times_param to be a float or int, got {gen_times_param}"
                )
                end = gen_times_param
            else:
                end = 0
            logger.warning(f"Using [<starting time> -> {end}] (inverted order!) evenly spaced times for regeneration")
            if isinstance(starting_video_time, Tensor) and torch.any(starting_video_time <= end).item():
                logger.error(
                    f"Some starting video times at inversion are smaller than the end time {end}: {starting_video_time[starting_video_time <= end]}"
                )
            video_times = np.linspace(  # (nb_video_timesteps,) or (nb_video_timesteps, nb_generated_samples)
                starting_video_time.float().numpy(force=True)  # expected to be close to 1
                if isinstance(starting_video_time, Tensor)
                else starting_video_time,
                end,
                eval_strat.nb_video_timesteps,
                endpoint=True,
            )
        case _ if isinstance(gen_times_type, list):
            logger.warning("Using provided list of video times for regeneration")
            assert len(gen_times_type) == eval_strat.nb_video_timesteps, (
                f"Expected nb_video_timesteps={eval_strat.nb_video_timesteps} video times, got {len(gen_times_type)}"
            )
            assert all(isinstance(t, float | int) for t in gen_times_type), (
                f"Expected gen_times_type to be a list of floats or ints, got {gen_times_type}"
            )
            video_times = np.array(gen_times_type)
        case _:
            raise ValueError(
                f"Unsupported gen_times_type: {gen_times_type}; expected 'evenly_spaced', 'evenly_spaced_from_inversion', or a list of floats"
            )
    logger.info(f"Using video times for regeneration: {video_times}")
    video_times = torch.tensor(video_times, device=cfg.device, dtype=cfg.dtype)
    if video_times.ndim == 1:
        video_times = video_times.unsqueeze(1).repeat(1, eval_strat.nb_generated_samples)
    # video_times is now of shape (nb_video_timesteps, nb_generated_samples)
    assert video_times.shape == (
        eval_strat.nb_video_timesteps,
        eval_strat.nb_generated_samples,
    ), f"Expected shape {(eval_strat.nb_video_timesteps, eval_strat.nb_generated_samples)}, got {video_times.shape}"

    # filter generated times if asked
    if eval_strat.selected_times is not None:
        with open(eval_strat.selected_times, "rb") as f:
            selected_times = pickle.load(f)
        assert isinstance(selected_times, list), f"Expected selected times to be a list, got {type(selected_times)}"
        assert isinstance(selected_times[0], tuple) and len(selected_times[0]) == 2, (
            f"Expected selected times to be a list of 2-tuples, got {type(selected_times[0])} at index 0"
        )
        assert isinstance(selected_times[0][0], float), (
            f"Expected selected times to be a list of 2-tuples of floats, got {type(selected_times[0][0])} at index 0"
        )
        logger.warning(
            f"Loaded selected times: {[(round(float(start), 2), round(float(end), 2)) for start, end in selected_times]}"
        )

        def time_in_selected_ranges(time_val: float):
            return any(start <= time_val <= end for start, end in selected_times)

        for video_time_matrix in video_times:  # video_time_matrix: (nb_generated_samples,)
            # we put NaN in video_times when the time for this frame for this video cell
            # is not in the selected ranges
            in_range_mask = torch.tensor([time_in_selected_ranges(t.item()) for t in video_time_matrix])
            video_time_matrix[~in_range_mask] = float("nan")

    video = []
    nb_vid_batches = ceil(len(video_times) / eval_strat.nb_video_times_in_parallel)

    video_time_pbar = pbar_manager.counter(
        total=nb_vid_batches,
        position=1,
        desc="Video timesteps batches",
        leave=False,
    )
    video_time_pbar.refresh()

    for vid_batch_idx in range(nb_vid_batches):
        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps" + " " * 4,
            leave=False,
        )
        diff_time_pbar.refresh()

        start = vid_batch_idx * eval_strat.nb_video_times_in_parallel
        end = (vid_batch_idx + 1) * eval_strat.nb_video_times_in_parallel
        video_time_batch = video_times[start:end]  # (nb_video_times_in_parallel, nb_generated_samples)
        this_nb_t_parallel = video_time_batch.shape[0]
        logger.debug(
            f"Processing {this_nb_t_parallel} video times (from {start + 1} to {start + this_nb_t_parallel}) out of {eval_strat.nb_video_timesteps}"
        )
        video_time_enc = []
        for video_time in video_time_batch:  # video_time_encoder only takes 1D tensors!
            # Nan in -> Nan out TODO: do not rely on this brittle behavior!
            enc = video_time_encoder.forward(video_time)
            video_time_enc.append(enc)
        video_time_enc = torch.cat(video_time_enc)
        # video_time_enc: (nb_video_times_in_parallel * nb_generated_samples, tim_embed_dim)

        image = torch.cat([inverted_gauss.clone() for _ in range(this_nb_t_parallel)])
        # image: (nb_video_times_in_parallel * nb_generated_samples, C, H, W)

        # video_time_enc has NaNs on dim 1: these are times that must be skipped (works per video cell)
        is_nan_time_cell = video_time_enc.isnan().any(dim=1)
        # is_nan_time_cell shape: (nb_video_times_in_parallel * nb_generated_samples,)
        mask = ~is_nan_time_cell

        # handle entirely skipped batches of times to avoid calling `net.forward` on empty data (errors)
        if not mask.any():
            logger.warning(
                f"All video cells are skipped at time batch {start + 1} to {start + this_nb_t_parallel}: skipping frame(s)"
            )
            # just put all to nan and let the logic below handle it
            image = torch.full_like(image, float("nan"))
        else:
            for t in inference_scheduler.timesteps:  # perform diffusion denoising on unmasked cells only
                model_output: torch.Tensor = (
                    net.forward(  # maybe here also NaNs in -> NaNs out? multiple dims though...
                        image[mask],
                        t,
                        encoder_hidden_states=video_time_enc[mask].unsqueeze(1),
                        return_dict=False,
                    )[0]
                )
                image[mask] = inference_scheduler.step(model_output, int(t), image[mask], return_dict=False)[0]
                diff_time_pbar.update()
            image[~mask] = torch.full_like(image[~mask], float("nan"))  # fill masked cells with nans

        diff_time_pbar.close()

        # Remove nan times per cell by copying the last non-nan frame
        # we need the immediatly previous frame to be able to propagate it, so we "add" it to the current batch for ease of processing
        if len(video) > 0:
            prev_frame = video[-1][-eval_strat.nb_generated_samples :]  # (nb_generated_samples, C, H, W)
        else:  # at start, just use a tensor of zeros (ie -1)
            prev_frame = torch.full(
                (eval_strat.nb_generated_samples, *image.shape[1:]),
                -1,
                device=image.device,
            )
        prev_frame_and_current_batch = torch.cat([prev_frame, image])  # this copies data!
        # prev_frame_and_current_batch: ((nb_video_times_in_parallel+1) * nb_generated_samples, C, H, W)
        # now let's propagate the last non-nan frame for all (batchified) samples accross all (batchified) times ... 😮‍💨
        for this_time_start_idx in range(  # we do it iteratively on the times inside this batch
            eval_strat.nb_generated_samples,
            prev_frame_and_current_batch.shape[0],
            eval_strat.nb_generated_samples,
        ):  # these are the indexes at which a new time starts in this batch, excluding the previous frame we added just above
            this_time_idxes = slice(
                this_time_start_idx,
                this_time_start_idx + eval_strat.nb_generated_samples,
            )
            prev_time_idxes = slice(
                this_time_start_idx - eval_strat.nb_generated_samples,
                this_time_start_idx,
            )
            # get nan cells at this time
            this_time = prev_frame_and_current_batch[this_time_idxes]  # (nb_generated_samples, C, H, W)
            nan_cells_mask = this_time.isnan()
            # modify in-place prev_frame_and_current_batch by propagating the previous frame on the nan cells
            this_time[nan_cells_mask] = prev_frame_and_current_batch[prev_time_idxes][nan_cells_mask]
            # because we run through the batch *iteratively*, we just need to propagate the immediatly previous frame:
            # if it was zeros, then all frames before it were also zeros; otherwise the cell will never be zero again
        # now let's update the original image tensor
        image = prev_frame_and_current_batch[eval_strat.nb_generated_samples :]
        video.append(image)
        video_time_pbar.update()

    video_time_pbar.close()

    # video is a list of ceil(nb_video_timesteps / nb_video_times_in_parallel) elements, each of shape
    # (nb_video_times_in_parallel * nb_generated_samples, C, H, W)
    video = torch.cat(video)  # (nb_video_timesteps * nb_generated_samples, C, H, W)
    video = video.split(eval_strat.nb_generated_samples)
    video = torch.stack(video)
    # save_images_or_videos expects (video_time, nb_generated_samples, C, H, W)
    expected_video_shape = (
        eval_strat.nb_video_timesteps,
        eval_strat.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    if video.shape != expected_video_shape:
        logger.error(f"Expected video shape {expected_video_shape}, got {video.shape}")
    logger.debug(f"Saving video tensor of shape {video.shape}")
    video_times = video_times.float().numpy(force=True)
    # two separate calls here to only save the individual frames of the -1_1 raw version
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
        texts=video_times,
    )
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["-1_1 raw"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
        also_save_individual_frames=True,
        also_save_individual_videos_separately=eval_strat.also_save_individual_videos_separately,
    )


def iterative_inverted_regeneration(
    cfg: InferenceConfig,
    eval_strat: IterativeInvertedRegeneration,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    inference_scheduler: SchedulersCommonClass,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
    true_label_time_0: str,
    config_snapshot: str,
    config_filename: str,
):
    """
    This strategy performs iteratively:
        1. an inversion to obtain the starting Gaussian
        2. a generation from that inverted Gaussian sample to obtain the next image of the video
    over all video timesteps.

    It is thus quite costly to run...
    """
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name

    clean_inference_strategy_folder(base_save_path, logger, config_snapshot, config_filename)

    # 0. Setup schedulers
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using scheduler:\n{inference_scheduler}")
    # thresholding is not supported by DDIMInverseScheduler; use "raw" clipping instead
    if inference_scheduler.config["thresholding"]:
        logger.warning("Using clipping instead of thresholding for inverted scheduler")
        kwargs = {"clip_sample": True, "clip_sample_range": 1}
    else:
        kwargs = {}
    inverted_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(inference_scheduler.config, **kwargs)  # pyright: ignore[reportAssignmentType]
    inverted_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using inverted scheduler config: {inverted_scheduler.config}")

    # 0.5. Get the starting batch
    batch, _ = get_starting_batch(
        cfg,
        eval_strat,
        logger,
        cfg.dataset.path,
        cfg.device,
        cfg.dtype,
        true_label_time_0,
        base_save_path,
    )

    # 1. Save the to-be noised images
    save_grid_of_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
    )
    save_histogram(batch, base_save_path / "starting_samples_histogram.png", (-1, 1), 50)

    # 2. Generate the trajectory
    video = []

    video_time_pbar = pbar_manager.counter(
        total=eval_strat.nb_video_timesteps,
        position=1,
        desc="Video timesteps",
        leave=False,
    )
    video_time_pbar.refresh()

    video_times = torch.linspace(0, 1, eval_strat.nb_video_timesteps, device=cfg.device, dtype=cfg.dtype)

    prev_video_time = 0
    image = batch
    for video_t_idx, video_time in enumerate(video_times):
        logger.debug(f"Video timestep index {video_t_idx + 1} / {eval_strat.nb_video_timesteps}")

        # 2.A Generate the inverted Gaussians
        inverted_gauss = image
        inversion_video_time = video_time_encoder.forward(prev_video_time, batch.shape[0])

        diff_time_pbar = pbar_manager.counter(
            total=len(inverted_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps (inversion)",
            leave=False,
        )
        diff_time_pbar.refresh()

        for t in inverted_scheduler.timesteps:
            model_output = net.forward(
                inverted_gauss,
                t,
                encoder_hidden_states=inversion_video_time.unsqueeze(1),
                return_dict=False,
            )[0]
            inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()

        save_grid_of_images_or_videos(
            inverted_gauss,
            base_save_path,
            f"inverted_gaussians_time{video_t_idx}",
            ["image min-max"],
            eval_strat.n_rows_displayed,
            0 if eval_strat.plate_name_to_simulate is not None else PADDING,
            logger,
        )
        save_histogram(
            inverted_gauss,
            base_save_path / f"inverted_gaussians_histogram_time{video_t_idx}.png",
            (-5, 5),
        )

        # 2.B Generate the next image from it
        image = inverted_gauss
        video_time_enc = video_time_encoder.forward(video_time.item(), batch.shape[0])

        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps (generation)",
            leave=False,
        )
        diff_time_pbar.refresh()

        for t in inference_scheduler.timesteps:
            model_output: torch.Tensor = net.forward(
                image,
                t,
                encoder_hidden_states=video_time_enc.unsqueeze(1),
                return_dict=False,
            )[0]
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()

        video.append(image.clone())
        save_histogram(
            image,
            base_save_path / f"image_histogram_time{video_t_idx}.png",
            (-1, 1),
            50,
        )

        prev_video_time = video_time.item()
        video_time_pbar.update()

    video_time_pbar.close(clear=True)

    # video is a list of nb_video_timesteps elements, each of shape (batch_size, channels, hight, width)
    video = torch.stack(video)  # (nb_video_timesteps, batch_size, channels, height, width)
    # save_images_or_videos expects (video_time, batch_size, channels, height, width)
    expected_video_shape = (
        eval_strat.nb_video_timesteps,
        eval_strat.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else PADDING,
        logger,
    )


def metrics_computation(
    cfg: InferenceConfig,
    eval_strat: MetricsComputation,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    inference_scheduler: SchedulersCommonClass,
    timesteps: list[float],
    timesteps2classnames: dict[float, str],
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
    accelerator: Accelerator,
    training_was_with_unpaired_data: bool,
    config_snapshot: str,
    config_filename: str,
):
    """
    Compute metrics such as FID.

    Everything is distributed.

    Adapted from utils/training.py
    """
    logger.info(f"----- Starting {eval_strat.name} -----\n{eval_strat}")
    logger.debug(
        f"Starting {eval_strat.name} on process ({accelerator.process_index})",
        main_process_only=False,
    )

    ##### 0. Preparations
    # cast models and tensors to eval strategy dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    eval_strat_dtype = dtype_map[eval_strat.dtype]
    logger.warning(f"Casting inference models to {eval_strat.dtype}")
    net = net.to(eval_strat_dtype)  # pyright: ignore[reportArgumentType]
    video_time_encoder = video_time_encoder.to(eval_strat_dtype)  # pyright: ignore[reportArgumentType]

    # Set precision flags
    torch.backends.fp32_precision = "ieee"  # pyright: ignore[reportAttributeAccessIssue]
    # also set the old flags, otherwise inductor fails
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Setup schedulers
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using scheduler:\n{inference_scheduler}")

    # Misc.
    metrics_computation_folder = cfg.output_dir / eval_strat.name
    if accelerator.is_main_process:
        # ignore metrics files from *other* feature extractors * comparison mode because we might run multiple metrics computations with different extractors / comparison modes
        names_to_ignore = [  # this needs to be synced with saving at the end of the func
            f.name
            for f in metrics_computation_folder.iterdir()
            if "metrics_dict" in f.name
            and not f.name.startswith(eval_strat.feature_extractor + "_" + eval_strat.metrics_comparison_mode)
        ]
        if not eval_strat.regen_images:
            names_to_ignore += [str(t) for t in eval_strat.selected_times]
            logger.warning(f"Reusing existing images from previous run: ignoring {names_to_ignore}")
        clean_inference_strategy_folder(
            metrics_computation_folder, logger, config_snapshot, config_filename, names_to_ignore
        )
    accelerator.wait_for_everyone()

    # use training time encodings
    assert list(timesteps2classnames.keys()) == timesteps, (
        f"Expected timesteps to be the same as keys in timesteps2classnames, got {timesteps} and {timesteps2classnames.keys()}"
    )
    classnames2timesteps = {v: k for k, v in timesteps2classnames.items()}
    eval_video_times = [classnames2timesteps[str(eval_time)] for eval_time in eval_strat.selected_times]
    if cfg.dataset.fully_ordered:
        eval_base_video_times_true_labels = [str(eval_time) for eval_time in eval_strat.selected_times]
        train_split_path = cfg.output_dir.parent / "train_samples.parquet"
        test_split_path = cfg.output_dir.parent / "test_samples.parquet"
        required_cols = {"true_label", "time"}
        if not eval_strat.load_times_from_base_parquet_file and train_split_path.exists() and test_split_path.exists():
            logger.info(
                f"Found split parquet files at {train_split_path} and {test_split_path}, loading them to get continuous video times for metrics computation"
            )
            train_df = pd.read_parquet(train_split_path)
            test_df = pd.read_parquet(test_split_path)
            assert required_cols.issubset(train_df.columns) and required_cols.issubset(test_df.columns), (
                f"Expected columns {required_cols} in split parquet files, got {train_df.columns} and {test_df.columns}"
            )
            all_files = pd.concat([train_df, test_df], ignore_index=True)
        else:
            logger.debug(
                f"eval_strat.load_times_from_base_parquet_file is False or split parquet files not found at {train_split_path} and {test_split_path}"
            )
            logger.info(
                f"Loading continuous video times for metrics computation from base parquet file at {cfg.dataset.path_to_single_parquet}"
            )
            assert cfg.dataset.path_to_single_parquet is not None, (
                "Expected path_to_single_parquet to be set in config for fully ordered dataset"
            )
            all_files = pd.read_parquet(cfg.dataset.path_to_single_parquet)
            assert required_cols.issubset(all_files.columns), (
                f"Expected columns {required_cols} in base parquet file, got {all_files.columns}"
            )
        logger.info(f"Loaded {len(all_files):,} samples time info for metrics computation")
        eval_video_time_preds = {
            true_time_label: all_files.loc[all_files["true_label"] == true_time_label, "time"].to_numpy()
            for true_time_label in eval_base_video_times_true_labels
        }
        logger.info(
            f"Fully ordered inference: selected continuous video times for metrics computation corresponding to continuous predictions from times {eval_strat.selected_times}"
        )
        for true_time_label, time_preds in eval_video_time_preds.items():
            logger.debug(
                f"Video time {true_time_label} has {len(time_preds)} time predictions: mean={time_preds.mean()}, std={time_preds.std()}"
            )
        eval_video_time_enc: Tensor | dict[str, Tensor] = {
            true_time_label: video_time_encoder.forward(
                torch.tensor(time_preds).to(accelerator.device, eval_strat_dtype)
            )
            for true_time_label, time_preds in eval_video_time_preds.items()
        }
    else:
        logger.info(
            f"Non fully ordered dataset: using selected video times for metrics computation directly from config: {eval_video_times} from {eval_strat.selected_times}"
        )
        eval_video_time_enc = video_time_encoder.forward(
            torch.tensor(eval_video_times).to(accelerator.device, eval_strat_dtype)
        )

    _shapes = (
        eval_video_time_enc.shape
        if isinstance(eval_video_time_enc, Tensor)
        else {k: v.shape for k, v in eval_video_time_enc.items()}
    )
    logger.debug(f"Shape of video time encodings for metrics computation: {_shapes}")

    # get true datasets to compare against (in [0; 255] uint8 PNG images)
    assert cfg.dataset.dataset_params is not None
    if cfg.dataset.dataset_params.dataset_class is ContinuousTimeImageDataset:
        discrete_ds_class = ImageDataset
    elif cfg.dataset.dataset_params.dataset_class is ContinuousTimeImageDataset1D:
        discrete_ds_class = ImageDataset1Dto3D
    else:
        raise ValueError(
            f"Unsupported dataset_params type {type(cfg.dataset.dataset_params)}; expected ContinuousTimeImageDataset or ContinuousTimeImageDataset1D"
        )

    true_datasets_to_compare_with = get_true_datasets_for_metrics_computation(
        eval_strat,
        eval_video_times,
        discrete_ds_class,
        cfg.dataset.dataset_params.file_extension,
        cfg.dataset.transforms,
        cfg,
        logger,
        timesteps2classnames,
        training_was_with_unpaired_data,
        metrics_computation_folder,
        eval_strat_dtype,
    )

    logger.info(f"Metrics comparison mode: {eval_strat.metrics_comparison_mode}")

    use_generated_candidate = eval_strat.metrics_comparison_mode.startswith("gen_vs_true")

    ##### 1. Generate the samples (if needed)
    if use_generated_candidate:
        if eval_strat.regen_images:
            _generate_images_for_metrics_computation(
                pbar_manager,
                eval_video_times,
                accelerator,
                eval_video_time_enc,
                eval_strat,
                timesteps2classnames,
                true_datasets_to_compare_with,
                metrics_computation_folder,
                training_was_with_unpaired_data,
                inference_scheduler,
                net,
                cfg,
                logger,
                eval_strat_dtype,
            )
        else:
            logger.warning("Skipping image generation for metrics computation")
    else:
        logger.info("Skipping image generation for metrics computation in true_vs_true mode")

    ##### 2. Compute metrics
    # TODO: weight tasks by number of samples
    # TODO: differentiate between using seen data only and all the available dataset!
    eval_time_tasks = [timesteps2classnames[eval_time] for eval_time in eval_video_times]

    # replace the "same time" true reference dataset with "all other times" if applicable
    use_other_times_reference = eval_strat.metrics_comparison_mode.endswith("all_other_times")
    if use_other_times_reference and not cfg.dataset.fully_ordered:
        raise ValueError(
            f"{eval_strat.metrics_comparison_mode} is currently implemented for fully ordered datasets only; falling back to same-time reference"
        )
    if use_other_times_reference and len(eval_time_tasks) < 2:
        raise ValueError(
            f"{eval_strat.metrics_comparison_mode} requires at least two selected times; falling back to same-time reference"
        )

    true_reference_datasets = true_datasets_to_compare_with
    if use_other_times_reference:
        logger.info("Using true reference datasets from all other selected times")
        true_reference_datasets = true_datasets_to_compare_with.copy()
        for task in eval_time_tasks:
            other_times_samples = [
                sample
                for other_task in eval_time_tasks
                if other_task != task
                for sample in true_datasets_to_compare_with[other_task].samples
            ]
            assert len(other_times_samples) > 0, (
                f"Expected at least one sample in other-times reference dataset for task {task}, got 0"
            )
            this_task_ds = true_datasets_to_compare_with[task]
            true_reference_datasets[task] = type(this_task_ds)(
                samples=other_times_samples,
                transforms=this_task_ds.transforms,
            )
            logger.debug(
                f"Built other-times true reference dataset for task {task}: {len(true_reference_datasets[task])} samples"
            )

    tasks = eval_time_tasks.copy()  # eval time names
    if eval_strat.also_compute_metrics_on_all_times:
        tasks += ["all_times"]
    # if "all_times" is present, dedicate a single rank to it and redistribute the rest
    if "all_times" in tasks and accelerator.num_processes > 1:
        all_times_rank = accelerator.num_processes - 1  # dedicate last rank to the heavy "all_times" task
        tasks_wo_all_times = [t for t in tasks if t != "all_times"]
        if accelerator.process_index == all_times_rank:
            tasks_for_this_process = ["all_times"]
        else:
            tasks_for_this_process = tasks_wo_all_times[accelerator.process_index :: accelerator.num_processes - 1]
    else:
        tasks_for_this_process = tasks[accelerator.process_index :: accelerator.num_processes]
    logger.debug(
        f"Tasks for process {accelerator.process_index}: {tasks_for_this_process}",
        main_process_only=False,
    )

    logger.info("Computing metrics...")
    metrics_dict: dict[str, dict[str, float]] = {}
    for task in tasks_for_this_process:
        if task == "all_times":
            reference_true_dataset = true_datasets_to_compare_with[task]
        else:
            reference_true_dataset = true_reference_datasets[task]

        # Optional cap for true reference data: randomly subsample without replacement.
        reference_true_dataset_for_metrics = reference_true_dataset
        if (
            eval_strat.max_nb_reference_samples is not None
            and len(reference_true_dataset) > eval_strat.max_nb_reference_samples
        ):
            sampled_reference_samples = random.sample(
                reference_true_dataset.samples,
                eval_strat.max_nb_reference_samples,
            )
            reference_true_dataset_for_metrics = type(reference_true_dataset)(
                samples=sampled_reference_samples,
                transforms=reference_true_dataset.transforms,
            )
            logger.info(
                f"Subsampled true reference data for task {task}: {len(reference_true_dataset)} -> {len(reference_true_dataset_for_metrics)} samples",
                main_process_only=False,
            )

        if use_generated_candidate:
            input2 = (
                metrics_computation_folder / task if task != "all_times" else metrics_computation_folder
            ).as_posix()
            nb_input2_samples = (
                len(list((metrics_computation_folder / task).iterdir()))
                if task != "all_times"
                else len(list(metrics_computation_folder.glob(f"*/*{cfg.dataset.dataset_params.file_extension}")))
            )
            input2_desc = input2
        else:
            input2 = true_datasets_to_compare_with[task]
            nb_input2_samples = len(input2)
            input2_desc = str(input2.base_path)

        nb_reference_samples = len(reference_true_dataset_for_metrics)
        if nb_input2_samples != nb_reference_samples:
            logger.warning(
                f"Mismatch in the number of samples for task {task}: {nb_input2_samples} candidate vs {nb_reference_samples} true reference",
                main_process_only=False,
            )
        logger.debug(
            f"Computing metrics on {nb_input2_samples} candidate samples from {input2_desc} vs {nb_reference_samples} true reference samples at {reference_true_dataset_for_metrics.base_path} on process {accelerator.process_index}",
            main_process_only=False,
        )
        # caching is used! Beware as I'm not sure how exactly does torch_fidelity handle it...
        metrics_kwargs = dict(
            input1=reference_true_dataset_for_metrics,
            input2=input2,
            cuda=True,
            batch_size=eval_strat.batch_size * 2,
            isc=False,
            fid=True,
            prc=True,
            kid=True,
            kid_subset_size=min(1_000, nb_reference_samples, nb_input2_samples),
            verbose=accelerator.is_main_process,
            cache=True,
            save_cpu_ram=task == "all_times" or use_other_times_reference,
            num_workers=8,
            feature_extractor=eval_strat.feature_extractor,
            feature_extractor_compile=cfg.compile,
        )
        if use_generated_candidate:
            metrics_kwargs["samples_find_deep"] = task == "all_times"
        metrics = torch_fidelity.calculate_metrics(**metrics_kwargs)
        metrics_dict[task] = metrics
        logger.debug(
            f"Computed metrics for time {task} on process {accelerator.process_index}: {metrics}",
            main_process_only=False,
        )

    # save this process' metrics to disk
    proc_metrics_path = (
        metrics_computation_folder
        / f"{eval_strat.feature_extractor}_{eval_strat.metrics_comparison_mode}_proc{accelerator.process_index}_metrics_dict.json"
    )
    with proc_metrics_path.open("w") as file:
        json.dump(metrics_dict, file)
    logger.debug(
        f"Saved metrics for process {accelerator.process_index} at {proc_metrics_path}",
        main_process_only=False,
    )
    accelerator.wait_for_everyone()

    ##### 3. Merge metrics from all processes & save them
    if accelerator.is_main_process:
        # get all proc files
        proc_metrics_files = [
            metrics_computation_folder
            / f"{eval_strat.feature_extractor}_{eval_strat.metrics_comparison_mode}_proc{proc_idx}_metrics_dict.json"
            for proc_idx in range(accelerator.num_processes)
        ]
        # merge them
        final_metrics_dict: dict[str, dict[str, float]] = {}

        logger.debug(f"Merging per-process metrics files: {[f.name for f in proc_metrics_files]}")
        for metrics_file in proc_metrics_files:
            with metrics_file.open("r") as file:
                proc_metrics = json.load(file)
            final_metrics_dict.update(proc_metrics)

        fname = f"{eval_strat.feature_extractor}_{eval_strat.metrics_comparison_mode}_all_procs_metrics_dict.json"
        with (metrics_computation_folder / fname).open("w") as file:
            json.dump(final_metrics_dict, file)
        logger.info(
            f"Saved metrics: {final_metrics_dict} to {metrics_computation_folder}/{fname}",
        )


def _generate_images_for_metrics_computation(
    pbar_manager: Manager,
    eval_video_times: list[float],
    accelerator: Accelerator,
    eval_video_time_enc: Tensor | dict[str, Tensor],
    eval_strat: MetricsComputation,
    timesteps2classnames: dict[float, str],
    true_datasets_to_compare_with: dict[str, BaseDataset],
    metrics_computation_folder: Path,
    training_was_with_unpaired_data: bool,
    inference_scheduler: SchedulersCommonClass,
    net: UNet2DConditionModel,
    cfg: InferenceConfig,
    logger: MultiProcessAdapter,
    eval_strat_dtype: torch.dtype,
):
    assert cfg.dataset.dataset_params is not None

    # loop over training video times
    video_times_pbar = pbar_manager.counter(
        total=len(eval_video_times),
        position=2,
        desc="Training video timesteps  ",
        enable=accelerator.is_main_process,
        leave=False,
    )
    if accelerator.is_main_process:
        video_times_pbar.refresh()

    for video_time_idx in video_times_pbar(range(len(eval_video_times))):
        video_time_idx: int
        if cfg.dataset.fully_ordered:
            true_label = str(eval_strat.selected_times[video_time_idx])
            video_time_enc = eval_video_time_enc[true_label]  # pyright: ignore[reportArgumentType]
        else:
            video_time_enc = eval_video_time_enc[video_time_idx].unsqueeze(0).repeat(eval_strat.batch_size, 1)  # pyright: ignore[reportArgumentType]

        # get timestep name
        video_time_name = timesteps2classnames[eval_video_times[video_time_idx]]

        logger.debug(f"Processing video time {video_time_name} with time encodings of shape {video_time_enc.shape}")

        # find how many samples to generate, batchify generation and distribute along processes
        gen_dir = metrics_computation_folder / video_time_name
        gen_dir.mkdir(parents=True, exist_ok=True)
        this_proc_gen_batches, tot_nb_samples = find_this_proc_this_time_batches_for_metrics_comp(
            eval_strat,
            len(true_datasets_to_compare_with[video_time_name]),
            logger,
            training_was_with_unpaired_data,
            accelerator,
        )
        logger.info(
            f"Process {accelerator.process_index} will generate batches of sizes: {this_proc_gen_batches} for time {video_time_name}, totalling {sum(this_proc_gen_batches)} samples out of {len(true_datasets_to_compare_with[video_time_name])} grand total true samples to compare to for this time",
            main_process_only=False,
        )
        if tot_nb_samples > len(video_time_enc):
            raise ValueError(
                f"Total number of samples to generate for this time ({tot_nb_samples}) is higher than the number of available video time encodings ({len(video_time_enc)})"
            )

        this_proc_idxes = list(range(accelerator.process_index, len(video_time_enc), accelerator.num_processes))

        batches_pbar = pbar_manager.counter(
            total=len(this_proc_gen_batches),
            position=3,
            desc="Evaluation batch" + 10 * " ",
            enable=accelerator.is_main_process,
            leave=False,
        )
        if accelerator.is_main_process:
            batches_pbar.refresh()

        # loop over generation batches
        for batch_idx, batch_size in batches_pbar(enumerate(this_proc_gen_batches)):
            if cfg.dataset.fully_ordered:
                start_on_this_proc_idxes = sum(this_proc_gen_batches[:batch_idx])
                end_on_this_proc_idxes = start_on_this_proc_idxes + batch_size
                idxes_on_this_batch_video_time_enc = this_proc_idxes[start_on_this_proc_idxes:end_on_this_proc_idxes]
                this_batch_video_time_enc = video_time_enc[idxes_on_this_batch_video_time_enc]
                if batch_idx % (5 + accelerator.process_index) == 0:
                    logger.debug(
                        f"At generation batch index {batch_idx} and on proc {accelerator.process_index}: using time preds at local index between {start_on_this_proc_idxes} and {end_on_this_proc_idxes} resulting in global indexes {idxes_on_this_batch_video_time_enc[:10] + ['...'] + idxes_on_this_batch_video_time_enc[-10:]}... resulting in video time encodings of shape {this_batch_video_time_enc.shape}",
                        main_process_only=False,
                    )
            else:
                this_batch_video_time_enc = video_time_enc[:batch_size]
                if batch_idx % (5 + accelerator.process_index) == 0:
                    logger.debug(
                        f"At generation batch index {batch_idx} and on proc {accelerator.process_index}: using video time encodings of shape {this_batch_video_time_enc.shape}",
                        main_process_only=False,
                    )
            assert batch_size == this_batch_video_time_enc.shape[0], (
                f"Expected batch size {batch_size} time encodings, got {this_batch_video_time_enc.shape[0]}"
            )

            gen_pbar = pbar_manager.counter(
                total=len(inference_scheduler.timesteps),
                position=4,
                desc="Generating samples" + " " * 8,
                enable=accelerator.is_main_process,
                leave=False,
            )
            if accelerator.is_main_process:
                gen_pbar.refresh()

            # generate a batch of samples
            image = torch.randn(
                batch_size,
                accelerator.unwrap_model(net).config["in_channels"],
                accelerator.unwrap_model(net).config["sample_size"],
                accelerator.unwrap_model(net).config["sample_size"],
                device=accelerator.device,
                dtype=eval_strat_dtype,
            )

            # loop over diffusion timesteps
            for t in gen_pbar(inference_scheduler.timesteps):
                model_output: torch.Tensor = net.forward(
                    image, t, this_batch_video_time_enc.unsqueeze(1), return_dict=False
                )[0]  # pyright: ignore[reportArgumentType]
                image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]

            # convert to f32 to avoid overflows
            image = image.to(torch.float32)

            # save to [0; 255] uint8 PNG RGB images
            save_images_for_metrics_compute(
                image,
                gen_dir,
                cfg.dataset.dataset_params.file_extension,
                accelerator.process_index,
            )

            gen_pbar.close(clear=True)

        batches_pbar.close(clear=True)
        # wait for everyone at end of each time (should be enough to avoid timeouts)
        accelerator.wait_for_everyone()

    video_times_pbar.close(clear=True)
    # no need to wait here then
    logger.debug("Finished image generation in MetricsComputation")

    ##### 1.5 Augment the generated samples if applicable
    if (
        isinstance(eval_strat.nb_samples_to_gen_per_time, str) and "aug" in eval_strat.nb_samples_to_gen_per_time
    ) or isinstance(eval_strat.nb_samples_to_gen_per_time, tuple):
        logger.info(
            f"Augmenting generated samples {2 ** len(eval_strat.augmentations_for_metrics_comp)} times for metrics computation with augmentations: {eval_strat.augmentations_for_metrics_comp}"
        )
        # Partition generated subdirectories among processes
        subdirs = [metrics_computation_folder / str(video_time_name) for video_time_name in eval_strat.selected_times]
        assigned_subdirs = subdirs[accelerator.process_index :: accelerator.num_processes]
        n_workers_per_process = os.cpu_count() // accelerator.num_processes
        for subdir in assigned_subdirs:
            logger.debug(
                f"Process {accelerator.process_index} augmenting folder {subdir}",
                main_process_only=False,
            )
            # augment
            assert cfg.dataset.dataset_params is not None
            extension = cfg.dataset.dataset_params.file_extension
            hard_augment_dataset_all_square_symmetries(
                subdir,
                logger,
                extension,
                n_workers_per_process,
                eval_strat.augmentations_for_metrics_comp,
            )
            # check result
            assert (
                nb_elems := len(list((subdir).glob(f"*.{extension}")))
                % 2 ** len(eval_strat.augmentations_for_metrics_comp)
                == 0
            ), (
                f"Expected number of elements to be a multiple of {2 ** len(eval_strat.augmentations_for_metrics_comp)}, got:\n{nb_elems} in {subdir}"
            )
        logger.info("Finished augmenting generated samples for metrics computation")

    # wait for data augmentation to finish before returning
    accelerator.wait_for_everyone()


def save_grid_of_images_or_videos(
    tensor: torch.Tensor,
    base_save_path: Path,
    artifact_name: str,
    norm_methods: list[ACCEPTED_NORMALIZATIONS],
    nrows: int,
    padding: int,
    logger: MultiProcessAdapter,
    also_save_individual_frames: bool = False,
    texts: ndarray | None = None,
    save_tensor: bool = True,
    also_save_individual_videos_separately: bool = False,
):
    """
    Save a `tensor` of images or videos to disk in a grid of `nrows`×`nrows`.

    - `tensor`: `(B, C, H, W)` or `(T, B, C, H, W)`
    - `texts`: `(T, B)` (no image only case for now)
    """
    # Save some raw images / trajectories to disk
    if save_tensor:
        file_path = base_save_path / f"{artifact_name}.pt"
        torch.save(tensor.half().cpu(), file_path)
        logger.debug(f"Saved raw {artifact_name} of shape {tensor.shape} to {file_path.name}")

    normalized_elements = normalize_elements_for_logging(tensor, norm_methods)
    logger.debug(f"Normalized {artifact_name} with methods {norm_methods}")

    match tensor.ndim:
        case 5:  # videos
            for norm_method, normed_vids in normalized_elements.items():
                # torch.save the videos in a grid
                save_path = base_save_path / f"{artifact_name}_{norm_method}.mp4"
                _save_grid_of_videos(
                    normed_vids,
                    save_path,
                    nrows,
                    padding,
                    logger,
                    also_save_individual_frames,
                    texts,
                    also_save_individual_videos_separately,
                )
                logger.debug(f"Saved {norm_method} normalized {artifact_name} to {save_path.name}")
        case 4:  # images
            for norm_method, normed_imgs in normalized_elements.items():
                # torch.save the images in a grid
                save_path = base_save_path / f"{artifact_name}_{norm_method}.png"
                _save_grid_of_images(normed_imgs, save_path, nrows, padding, logger, texts)
                logger.debug(f"Saved {norm_method} normalized {artifact_name} to {save_path.name}")
        case _:
            raise ValueError(f"Expected 4D or 5D tensor, got {tensor.ndim} with shape {tensor.shape}")

    return save_path  # pyright: ignore[reportPossiblyUnboundVariable]


def _save_grid_of_videos(
    videos_tensor: ndarray,
    save_path: Path,
    nrows: int,
    padding: int,
    logger: MultiProcessAdapter,
    also_save_individual_frames: bool = False,
    texts: ndarray | None = None,
    also_save_individual_videos_separately: bool = False,
):
    """
    - `videos_tensor`: `(T, B, C, H, W)`
    - `texts`: `(T, B)`
    """
    # Checks
    assert videos_tensor.ndim == 5, f"Expected 5D tensor, got {videos_tensor.shape}"
    assert videos_tensor.dtype == np.uint8, f"Expected dtype uint8, got {videos_tensor.dtype}"
    assert videos_tensor.min() >= 0 and videos_tensor.max() <= 255, (
        f"Expected [0;255] range, got [{videos_tensor.min()}, {videos_tensor.max()}]"
    )
    if videos_tensor.shape[1] != nrows**2:
        logger.warning(
            f"Expected nrows²={nrows**2} videos at dim index 1, got shape: {videos_tensor.shape}. Selecting first {nrows**2} videos."
        )
        videos_tensor = videos_tensor[:, : nrows**2]
    if texts is not None:
        assert texts.shape == videos_tensor.shape[:2], (
            f"Expected texts of shape {videos_tensor.shape[:2]}, got {texts.shape}"
        )

    # Convert tensor to a grid of videos
    fps = max(1, int(len(videos_tensor) / 10))
    logger.debug(f"Using fps {fps}")
    writer = imageio.get_writer(
        save_path,
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv444p",
        ffmpeg_params=[
            "-crf",
            "0",
            "-preset",
            "veryslow",
        ],
    )

    if also_save_individual_frames:
        (save_path.parent / save_path.stem).mkdir(exist_ok=True)

    if also_save_individual_videos_separately:
        for vid_idx in range(videos_tensor.shape[1]):
            (save_path.parent / save_path.stem / f"video_{vid_idx}").mkdir(exist_ok=True)

    for frame_idx, frame in enumerate(videos_tensor):
        # save each image to the vid subfolder if requested
        if also_save_individual_videos_separately:
            for vid_idx, vid in enumerate(frame):
                img_save_path = save_path.parent / save_path.stem / f"video_{vid_idx}" / f"frame{frame_idx}.png"
                Image.fromarray(vid).save(img_save_path)

        # make grid at that time
        grid_img = make_grid(torch.from_numpy(frame), nrow=nrows, padding=padding)
        # to np array
        np_img = grid_img.numpy().transpose(1, 2, 0)
        # if provided, write text on that frame
        if texts is not None:
            pil_img = Image.fromarray(np_img)
            draw = ImageDraw.Draw(pil_img)
            h, w = frame.shape[-2:]  # height and width of one video cell
            for cell_idx in range(frame.shape[0]):
                row = cell_idx // nrows
                col = cell_idx % nrows
                x = col * (w + padding) + padding // 2
                y = row * (h + padding) + padding // 2
                t_val = texts[frame_idx, cell_idx]
                if np.isnan(t_val):
                    t_text = "skipped"
                else:
                    t_text = f"{t_val:.2f}"
                draw.rectangle([x + 5, y + 5, x + 25, y + 15], fill=(0, 0, 0))
                draw.text((x + 5, y + 5), t_text, fill=(255, 255, 255))
            np_img = np.array(pil_img)
        # add to writer
        logger.debug(f"Adding frame {frame_idx + 1}/{len(videos_tensor)} | shape: {np_img.shape}")
        writer.append_data(np_img)
        # save individual frames if requested
        if also_save_individual_frames:
            frame_path = save_path.parent / save_path.stem / f"frame_{frame_idx}.png"
            Image.fromarray(np_img).save(frame_path)
            logger.debug(f"Saved frame {frame_idx + 1}/{len(videos_tensor)} at {frame_path}")

    writer.close()


def _save_grid_of_images(
    images_tensor: ndarray,
    save_path: Path,
    nrows: int,
    padding: int,
    logger: MultiProcessAdapter,
    texts: ndarray | None = None,
):
    # Checks
    assert images_tensor.ndim == 4, f"Expected 4D tensor, got {images_tensor.shape}"
    assert images_tensor.dtype == np.uint8, f"Expected dtype uint8, got {images_tensor.dtype}"
    assert images_tensor.min() >= 0 and images_tensor.max() <= 255, (
        f"Expected [0;255] range, got [{images_tensor.min()}, {images_tensor.max()}]"
    )
    if texts is not None:
        assert texts.ndim == 1, f"Expected 1D texts, got {texts.shape}"
        assert texts.shape[0] == images_tensor.shape[0], (
            f"Expected texts of shape {images_tensor.shape[0]}, got {texts.shape}"
        )
    if images_tensor.shape[0] != nrows**2:
        logger.warning(
            f"Expected nrows²={nrows**2} images, got {images_tensor.shape[0]}. Selecting first {nrows**2} images."
        )
        images_tensor = images_tensor[: nrows**2]

    # Convert tensor to a grid of images
    grid_img = make_grid(torch.from_numpy(images_tensor), nrow=nrows, padding=padding)
    # to np array
    grid_img = grid_img.numpy().transpose(1, 2, 0)
    # if provided, write text on that frame
    if texts is not None:
        pil_img = Image.fromarray(grid_img)
        draw = ImageDraw.Draw(pil_img)
        h, w = images_tensor.shape[2:]  # height and width of one image cell
        for cell_idx in range(images_tensor.shape[0]):
            row = cell_idx // nrows
            col = cell_idx % nrows
            x = col * (w + padding) + padding // 2
            y = row * (h + padding) + padding // 2
            t_val = texts[cell_idx]
            if t_val == float("nan"):
                t_text = "skipped"
            else:
                t_text = f"{t_val:.2f}"
            draw.rectangle([x + 5, y + 5, x + 25, y + 15], fill=(0, 0, 0))
            draw.text((x + 5, y + 5), t_text, fill=(255, 255, 255))
        grid_img = np.array(pil_img)

    # Convert to PIL Image
    pil_img = Image.fromarray(grid_img)
    pil_img.save(save_path)


def save_histogram(
    images_tensor: torch.Tensor,
    save_path: Path,
    xlims: tuple[float, float] | None = None,
    ymax: float | None = None,
):
    # Checks
    assert images_tensor.ndim == 4, f"Expected 4D tensor, got {images_tensor.shape}"
    assert images_tensor.shape[1] == 3, f"Expected 3 channels, got {images_tensor.shape[1]}"

    # Compute histograms per channel
    histograms = []
    for channel in range(3):
        counts, bins = np.histogram(images_tensor[:, channel].cpu().numpy().flatten(), bins=256, density=True)
        histograms.append((counts, bins))

    # Plot histograms
    plt.figure(figsize=(10, 6))
    colors = ("red", "green", "blue")
    for i, color in enumerate(colors):
        plt.stairs(histograms[i][0], histograms[i][1], color=color, label=f"{color} channel")
    plt.legend()
    plt.title("Histogram of Image Channels")
    suptitle = " ".join(word.capitalize() for word in save_path.stem.split("_"))
    plt.suptitle(suptitle)
    plt.xlabel("Pixel Value")
    plt.ylabel("Density")
    plt.grid()
    if xlims is not None:
        plt.xlim(xlims)
    if ymax is not None:
        plt.ylim((0, ymax))

    # Plot Gaussian distribution if "gaussian" in save_path
    xlims = plt.xlim()
    if "gaussian" in save_path.stem.lower():
        x = np.linspace(xlims[0], xlims[1], 256)
        plt.plot(
            x,
            stats.norm.pdf(x),
            color="grey",
            linestyle="dashed",
            label="Gaussian",
            alpha=0.6,
        )

    # Save plot
    plt.savefig(save_path)
    plt.close()
    print(f"Saved histogram to {save_path}")


def plot_side_by_side_comparison(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    base_save_path: Path,
    artifact1_name: str,
    artifact2_name: str,
    metric_name: str,
    norm_methods: list[ACCEPTED_NORMALIZATIONS],
    nrows: int,
    logger: MultiProcessAdapter,
):
    """
    Save a side-by-side comparison of two tensors of images or videos to disk in a grid of `nrows`×`nrows`,
    with 2 images / videos per cell
    """
    # Checks
    assert tensor1.shape == tensor2.shape, f"Expected same shape, got {tensor1.shape} and {tensor2.shape}"

    # Save some raw images / trajectories to disk
    base_save_path.mkdir(parents=True, exist_ok=True)
    torch.save(tensor1.half().cpu(), base_save_path / f"{artifact1_name}.pt")
    logger.debug(f"Saved raw {artifact1_name} of shape {tensor1.shape} to {base_save_path}/{artifact1_name}.pt")
    torch.save(tensor2.half().cpu(), base_save_path / f"{artifact2_name}.pt")
    logger.debug(f"Saved raw {artifact2_name} of shape {tensor2.shape} to {base_save_path}/{artifact2_name}.pt")

    normalized_t1 = normalize_elements_for_logging(tensor1, norm_methods)
    normalized_t2 = normalize_elements_for_logging(tensor2, norm_methods)

    match tensor1.ndim:
        case 5:  # videos
            raise NotImplementedError("TODO if needed")
        case 4:  # images
            for norm_method in normalized_t1:
                t1, t2 = normalized_t1[norm_method], normalized_t2[norm_method]
                # torch.save the images in a grid
                save_path = base_save_path / f"{metric_name}_{artifact1_name}_vs_{artifact2_name}_{norm_method}.png"
                _save_side_by_side_of_images(t1, t2, save_path, nrows, logger)
                logger.debug(
                    f"Saved {norm_method} normalized {artifact1_name} vs {artifact2_name} side-by-side to {save_path.name}"
                )
        case _:
            raise ValueError(f"Expected 4D or 5D tensor, got {tensor1.ndim} with shape {tensor1.shape}")

    return save_path  # pyright: ignore[reportPossiblyUnboundVariable]


def _save_side_by_side_of_images(
    t1: ndarray,
    t2: ndarray,
    save_path: Path,
    nrows: int,
    logger: MultiProcessAdapter,
):
    # Checks
    assert t1.ndim == t2.ndim == 4, f"Expected 4D tensor, got {t1.shape} and {t2.shape}"
    assert t1.shape == t2.shape, f"Expected same shape, got {t1.shape} and {t2.shape}"
    assert t1.dtype == t2.dtype == np.uint8, f"Expected dtype uint8, got {t1.dtype} and {t2.dtype}"
    assert t1.min() >= 0 and t1.max() <= 255, f"Expected [0;255] range, got [{t1.min()}, {t1.max()}]"
    assert t2.min() >= 0 and t2.max() <= 255, f"Expected [0;255] range, got [{t2.min()}, {t2.max()}]"
    if t1.shape[0] != nrows**2:
        logger.warning(f"Expected nrows²={nrows**2} images, got {t1.shape[0]}. Selecting first {nrows**2} images.")
        t1 = t1[: nrows**2]
        t2 = t2[: nrows**2]

    # interleave the two tensors
    interleaved_imgs = np.empty((t1.shape[0] * 2, *t1.shape[1:]), dtype=t1.dtype)
    interleaved_imgs[0::2] = t1
    interleaved_imgs[1::2] = t2

    # Convert tensor to a grid of images
    grid_img = make_grid(torch.from_numpy(interleaved_imgs), nrow=nrows * 2)

    # Convert to PIL Image
    pil_img = Image.fromarray(grid_img.numpy().transpose(1, 2, 0))
    pil_img.save(save_path)


def get_true_datasets_for_metrics_computation(
    eval_strat: MetricsComputation,
    eval_video_times: list[float],
    dataset_class: type[BaseDataset],
    data_files_common_suffix: str,
    test_transforms: Compose,
    cfg: InferenceConfig,
    logger: MultiProcessAdapter,
    timesteps2classnames: dict[float, str],
    training_was_with_unpaired_data: bool,
    metrics_computation_folder: Path,
    eval_strat_dtype: torch.dtype,
):
    """
    Return the true datasets to compare against for metrics computation, in [0; 255] uint8 tensors like saved generated images.

    Depending on `eval_strat.nb_samples_to_gen_per_time` the true datasets returned by this function can be:
    - the base datasets if `eval_strat.nb_samples_to_gen_per_time` is a number or does not contain 'aug'
    - the hard augmented versions if `"aug" in nb_samples_to_gen_per_time"`
    - half (taken at random) of whatever is used from last step if `"half" in nb_samples_to_gen_per_time`

    Copied/Adapted from `GaussianProxy/utils/training.py`.
    """
    # Misc.
    base_dataset_path = Path(cfg.dataset.path)
    eval_time_names = [timesteps2classnames[eval_time] for eval_time in eval_video_times]
    true_datasets_to_compare_with: dict[str, BaseDataset] = {}

    # Use the correct image processing ([0; 255] uint8) for metrics computation
    assert any(isinstance(t, Normalize_v1 | Normalize_v2) for t in test_transforms.transforms), (
        f"Expected normalization to be in test transforms, got : {test_transforms}"
    )
    nb_channels = cfg.dataset.data_shape[0]
    metrics_compute_transforms = Compose(
        [
            # 1: Process images for inference (== training processing \ augmentations)
            test_transforms,  # test transforms *must* include the normalization to [-1, 1]
            # 2: If the model is *inferring* in f16, bf16, ..., then also discretize the true samples to simulate the same processing!
            ToDtype(eval_strat_dtype, scale=True),  # no-op if already f32
            # 3: Convert back to f32 *before* scaling
            ToDtype(torch.float32, scale=True),  # no-op if already f32
            # 4: Scale back from [-1, 1] to [0, 1]
            Normalize_v2(mean=[-1] * nb_channels, std=[2] * nb_channels),
            # 5: Convert to [0; 255] uint8 for PIL png saving
            ToDtype(torch.uint8, scale=True),
        ]
    )
    logger.info(f"Using transforms for true datasets in metrics computation: {metrics_compute_transforms}")
    # TODO: programmatically check consistency with true samples processing in misc/save_images_for_metrics_compute
    base_aug_factor = 2 ** len(eval_strat.augmentations_for_metrics_comp)  # only used if 'aug' strategy
    logger.debug(
        f"Using augmentations: {eval_strat.augmentations_for_metrics_comp} giving base_aug_factor: {base_aug_factor} for 'aug' strategies"
    )

    # This is where we choose what to compare metrics against
    all_files_per_time = {}
    hard_aug_ds_path = None
    ds_path = None

    # First, get samples quickly for fully ordered datasets if not using augmented versions
    if cfg.dataset.fully_ordered and "aug" not in str(eval_strat.nb_samples_to_gen_per_time):
        train_split_path = cfg.output_dir.parent / "train_samples.parquet"
        test_split_path = cfg.output_dir.parent / "test_samples.parquet"
        required_cols = {"true_label", "file_path"}
        if train_split_path.exists() and test_split_path.exists():
            train_df = pd.read_parquet(train_split_path)
            test_df = pd.read_parquet(test_split_path)
            assert required_cols.issubset(train_df.columns) and required_cols.issubset(test_df.columns), (
                f"Expected columns {required_cols} in split parquet files, got {train_df.columns} and {test_df.columns}"
            )
            all_files = pd.concat([train_df, test_df], ignore_index=True)
            logger.debug(f"Using train/test split file enumeration from {train_split_path} and {test_split_path}")
        else:
            assert cfg.dataset.path_to_single_parquet is not None, (
                "Expected path_to_single_parquet in config to be able to enumerate files for fully_ordered dataset, got None."
            )
            all_files = pd.read_parquet(cfg.dataset.path_to_single_parquet)
            assert required_cols.issubset(all_files.columns), (
                f"Expected columns {required_cols} in single parquet file, got {all_files.columns}"
            )
        for time_name in eval_time_names:
            all_files_per_time[time_name] = [
                Path(path) for path in all_files.loc[all_files["true_label"] == time_name, "file_path"].to_list()
            ]
    # If using augmented versions, prepare to fall back to directory enumeration but on the hard aug version
    elif "aug" in str(eval_strat.nb_samples_to_gen_per_time) and "_hard_augmented" not in base_dataset_path.name:
        if cfg.dataset.hard_aug_path is not None:
            hard_aug_ds_path = Path(cfg.dataset.hard_aug_path)
        else:
            hard_aug_ds_path = base_dataset_path.with_name(base_dataset_path.name + "_hard_augmented")
        assert hard_aug_ds_path.exists(), (
            f"Expected existing hard augmented dataset at {hard_aug_ds_path} (remember it's also possible to set `dataset.hard_aug_path`...)"
        )
        ds_path = hard_aug_ds_path
        logger.info(f"Using hard augmented version at {ds_path}")
    # Otherwise (not fully ordered or not using augmented versions), also prepare to fall back to directory enumeration
    else:
        ds_path = base_dataset_path

    # helper
    def list_files(time_name: str, base_path: Path):
        current_path = base_path / time_name
        files = [f for f in current_path.iterdir() if f.is_file() and f.suffix == "." + data_files_common_suffix]
        return time_name, files

    # Retrieve files by directory enumeration if needed
    if len(all_files_per_time) == 0:
        assert ds_path is not None, (
            "Expected ds_path to be set for directory enumeration of true samples for metrics computation, got None"
        )

        logger.debug(
            f"Listing files in dataset path {ds_path} for time names {eval_time_names} to create true datasets for metrics computation"
        )
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(list_files, t, ds_path): t for t in eval_time_names}
            for future in as_completed(futures):
                time_name, files = future.result()
                all_files_per_time[time_name] = files

    # Then if generating half the number of samples, take half of the available true data to compare with
    if isinstance(eval_strat.nb_samples_to_gen_per_time, str) and "half" in eval_strat.nb_samples_to_gen_per_time:
        logger.debug("'half' in nb_samples_to_gen_per_time: comparing to half the true dataset")
        for time_name in all_files_per_time:
            all_files_per_time[time_name] = all_files_per_time[time_name][: len(all_files_per_time[time_name]) // 2]

    # Finally instantiate the datasets
    if cfg.debug:
        logger.debug("Instantiating small true datasets to compare with for metrics computation")
        for time_name, all_files in all_files_per_time.items():
            true_datasets_to_compare_with[time_name] = dataset_class(
                samples=all_files[:50],  # only 50 samples for debug
                transforms=metrics_compute_transforms,
            )
    else:
        for time_name, all_files in all_files_per_time.items():
            true_datasets_to_compare_with[time_name] = dataset_class(
                samples=all_files,
                transforms=metrics_compute_transforms,
            )

    # add the 'all_times' dataset if needed, but do not take the augmented version (well unless it's already the base one)
    if eval_strat.also_compute_metrics_on_all_times:
        logger.info(f"Also computing metrics on all times using base dataset at {base_dataset_path}")
        all_files_all_times = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(list_files, t, base_dataset_path): t for t in eval_time_names}
            for future in as_completed(futures):
                _, files = future.result()
                all_files_all_times.extend(files)
        if cfg.debug:
            all_files_all_times = all_files_all_times[:500]
        true_datasets_to_compare_with["all_times"] = dataset_class(
            samples=all_files_all_times,
            transforms=metrics_compute_transforms,
        )

    # Check that datasets are well-formed
    for time_name, dataset in true_datasets_to_compare_with.items():
        assert len(dataset) > 0, (
            f"No samples found for time {time_name} when creating true dataset to compare with for metrics computation"
        )
        if not cfg.debug and (
            (isinstance(eval_strat.nb_samples_to_gen_per_time, str) and "aug" in eval_strat.nb_samples_to_gen_per_time)
            or isinstance(eval_strat.nb_samples_to_gen_per_time, tuple)
        ):
            if "half" in eval_strat.nb_samples_to_gen_per_time:
                assert training_was_with_unpaired_data or len(dataset) % (base_aug_factor / 2) == 0, (
                    f"Expected number of samples to be a multiple of {base_aug_factor / 2} when using paired data and 'half', got {len(dataset)} for time {time_name} at {dataset.base_path}"
                )
            elif not (training_was_with_unpaired_data or len(dataset) % base_aug_factor == 0):
                logger.error(
                    f"Expected number of samples to be a multiple of {base_aug_factor} when using paired data and no 'half', got {len(dataset)} for time {time_name} at {dataset.base_path}"
                )
        logger.debug(
            f"True dataset to compare with for metrics computation for time {time_name} has {len(dataset)} samples at {dataset.base_path}"
        )

    # Save a few samples to visually check the processing!
    logger.info(
        f"Saving a few true samples for processing check at {metrics_computation_folder}/few_true_samples_for_processing_check"
    )
    for time, dataset in true_datasets_to_compare_with.items():
        few_samples_indexes = random.sample(range(len(dataset)), 5)
        few_samples = dataset.__getitems__(few_samples_indexes)
        for sample_idx, sample in enumerate(few_samples):
            pil_img = Image.fromarray(sample.permute(1, 2, 0).squeeze().numpy())
            orig_filename = dataset.samples[few_samples_indexes[sample_idx]].stem
            out_dir = metrics_computation_folder / "few_true_samples_for_processing_check" / time
            out_dir.mkdir(parents=True, exist_ok=True)
            pil_img.save(out_dir / f"{orig_filename}_processed.{data_files_common_suffix}")

    return true_datasets_to_compare_with


def find_this_proc_this_time_batches_for_metrics_comp(
    eval_strat: MetricsComputation,
    nb_true_data_class_samples: int,
    logger: MultiProcessAdapter,
    training_was_with_unpaired_data: bool,
    accelerator: Accelerator,
):
    """
    Return the list of batch sizes to generate, for a given `video_time_idx` and splitting between processes.

    `eval_strat.nb_samples_to_gen_per_time` can be:
    - `int`: the number of samples to generate
    - `"adapt"`: generate as many samples as there are in the true data time
    - `"adapt half"`: generate half as many samples as there are in the true data time
    - `"adapt aug"`: generate as many samples as there are in the true data time, then n⨉ augment them (up to full Dih4)
    - `"adapt half aug"`: generate half as many samples as there are in the true data time, then n⨉ augment them (up to full Dih4)
    - `tuple(int, 'aug')`: generate **the given number** of samples, _then_ n⨉ augment them (up to full Dih4)

    Copied/Adapted from `GaussianProxy/utils/training.py`.
    """
    # find total number of samples to generate for this video time
    base_aug_factor: int = 2 ** len(eval_strat.augmentations_for_metrics_comp)

    if isinstance(eval_strat.nb_samples_to_gen_per_time, int):
        tot_nb_samples = eval_strat.nb_samples_to_gen_per_time
    elif eval_strat.nb_samples_to_gen_per_time == "adapt":
        tot_nb_samples = nb_true_data_class_samples
    elif eval_strat.nb_samples_to_gen_per_time == "adapt half":
        tot_nb_samples = nb_true_data_class_samples // 2
    elif eval_strat.nb_samples_to_gen_per_time == "adapt aug":
        nb_all_aug_samples = nb_true_data_class_samples
        assert training_was_with_unpaired_data or nb_all_aug_samples % base_aug_factor == 0, (
            f"Expected number of samples to be a multiple of {base_aug_factor} when using paired data, got {nb_all_aug_samples}"
        )
        tot_nb_samples = nb_all_aug_samples // base_aug_factor  # base_aug_factor ⨉ augment them
        logger.debug(f"Will augment samples {base_aug_factor}⨉ after generation")
    elif eval_strat.nb_samples_to_gen_per_time == "adapt half aug":
        nb_all_aug_samples = nb_true_data_class_samples
        assert training_was_with_unpaired_data or nb_all_aug_samples % base_aug_factor == 0, (
            f"Expected number of samples to be a multiple of {base_aug_factor} when using paired data, got {nb_all_aug_samples}"
        )
        tot_nb_samples = nb_all_aug_samples // (
            base_aug_factor * 2
        )  # half the number of samples, *then* base_aug_factor ⨉ augment them
        logger.debug(f"Will augment samples {base_aug_factor}⨉ after generation")
    elif (
        isinstance(eval_strat.nb_samples_to_gen_per_time, tuple)
        and len(eval_strat.nb_samples_to_gen_per_time) == 2
        and isinstance(eval_strat.nb_samples_to_gen_per_time[0], int)
        and eval_strat.nb_samples_to_gen_per_time[1] == "aug"
    ):
        tot_nb_samples = eval_strat.nb_samples_to_gen_per_time[0]
        logger.debug(f"Will augment samples {base_aug_factor}⨉ after generation")
    else:
        raise ValueError(
            f"Expected 'nb_samples_to_gen_per_time' to be an int, 'adapt', 'adapt half', 'adapt aug', 'adapt half aug', or '(<int>, 'aug')' got {eval_strat.nb_samples_to_gen_per_time}"
        )

    # share equally among processes & batchify
    # the code below ensures all processes have the same number of batches to generate,
    # with the same number of samples in each batch but the last one
    # (with a diff of at most 1 for the last)
    this_proc_nb_full_batches, last_batch_to_share = divmod(
        tot_nb_samples, accelerator.num_processes * eval_strat.batch_size
    )
    this_proc_last_batch, remainder = divmod(last_batch_to_share, accelerator.num_processes)
    this_proc_gen_batches = [eval_strat.batch_size] * this_proc_nb_full_batches + [this_proc_last_batch]
    if accelerator.process_index < remainder:
        this_proc_gen_batches[-1] += 1
    # pop the last batch if it is empty
    if this_proc_gen_batches[-1] == 0:
        this_proc_gen_batches.pop()

    return this_proc_gen_batches, tot_nb_samples


def get_starting_batch(
    cfg: InferenceConfig,
    eval_strat: InvertedRegeneration
    | SimpleGeneration
    | ForwardNoising
    | InversionRegenerationOnly
    | VideoGenerationFromNoise,
    logger: MultiProcessAdapter,
    dataset_path: Path | str,
    device: torch.device | str,
    dtype: torch.dtype | str,
    true_label_time_0: str,
    base_save_path: Path,
):
    # build the starting dataset
    if cfg.dataset.fully_ordered:
        logger.warning_once("Using a fully ordered dataset with continuous starting times")
        return _get_starting_batch_continuous(cfg, eval_strat, logger, device, dtype, true_label_time_0, base_save_path)
    else:
        logger.warning_once("Using a discrete dataset with 0 (zero) as starting time")
        return _get_starting_batch_discrete(cfg, eval_strat, logger, dataset_path, device, dtype)


def _get_starting_batch_continuous(
    cfg: InferenceConfig,
    eval_strat: InvertedRegeneration
    | SimpleGeneration
    | ForwardNoising
    | InversionRegenerationOnly
    | VideoGenerationFromNoise,
    logger: MultiProcessAdapter,
    device: torch.device | str,
    dtype: torch.dtype | str,
    true_label_time_0: str,
    base_save_path: Path,
) -> tuple[Tensor, Tensor]:
    assert cfg.dataset.dataset_params is not None
    assert cfg.dataset.path_to_single_parquet is not None

    # build the starting dataset
    df = pd.read_parquet(cfg.dataset.path_to_single_parquet)
    # ensure the index of the df is integer indices to be able to use __getitems__ below
    df = df.reset_index(drop=True)
    # ensure no augmentations here
    kept_transforms = remove_flips_and_rotations_from_transforms(cfg.dataset.transforms)[0]
    starting_ds: BaseContinuousTimeDataset = cfg.dataset.dataset_params.dataset_class(
        df,
        kept_transforms,
        cfg.dataset.expected_initial_data_range,
    )
    logger.info(f"Built dataset:\n{starting_ds}")
    logger.info(f"Using transforms:\n{kept_transforms}")

    # get the starting batch
    if cfg.dataset.separate_gt_starting_class_path is not None:
        logger.warning_once(f"Using a separate ground truth path: {cfg.dataset.separate_gt_starting_class_path}")
        gt_stems: set[str] = set(  # TODO: HARDCODED FOR EPENDYMAL CUTOUT!!!
            [re.sub(r"_aug\d+$", "", f.stem) for f in Path(cfg.dataset.separate_gt_starting_class_path).iterdir()]
        )
        stems = df["file_path"].str.rsplit("/", n=1).str[-1].str.replace(r"_aug\d+.png$", "", regex=True)
        mask = stems.isin(gt_stems)
        all_train_files_discrete_time_0 = df.loc[mask]
    else:
        all_train_files_discrete_time_0 = df[df["true_label"] == true_label_time_0]
    with pd.option_context("max_colwidth", 200):
        logger.info(f"Using discrete-time-0 DataFrame:\n{all_train_files_discrete_time_0}")

    # select starting samples to generate from
    if hasattr(eval_strat, "plate_name_to_simulate") and eval_strat.plate_name_to_simulate is not None:  # pyright: ignore[reportAttributeAccessIssue]
        # raise NotImplementedError("Plate-based selection not implemented yet for continuous-time datasets; TODO!")
        sampled_train_files_time0 = all_train_files_discrete_time_0[
            all_train_files_discrete_time_0["file_path"].str.contains(eval_strat.plate_name_to_simulate)  # pyright: ignore[reportAttributeAccessIssue]
        ]
        assert len(sampled_train_files_time0) > 0, (
            f"No samples found with label {true_label_time_0} and plate name {eval_strat.plate_name_to_simulate}"  # pyright: ignore[reportAttributeAccessIssue]
        )
        logger.info(f"Found {len(sampled_train_files_time0)} patches from plate {eval_strat.plate_name_to_simulate}")  # pyright: ignore[reportAttributeAccessIssue]
        assert len(sampled_train_files_time0) == eval_strat.nb_generated_samples, (
            f"Expected to get {eval_strat.nb_generated_samples} samples, got {len(sampled_train_files_time0)}"
        )

        # sort samples in (x,y) dictionary order
        def xy_from_stem(pathlike: str | Path) -> tuple[int, int]:
            stem = Path(pathlike).stem
            m = re.compile(r".*_(\d+)_([0-9]+)$").match(stem)
            if not m:
                raise ValueError(f"Could not parse XY from stem '{stem}' (expected trailing '_X_Y').")
            return int(m.group(1)), int(m.group(2))

        sampled_train_files_time0 = sampled_train_files_time0.copy()
        xy_pairs = sampled_train_files_time0["file_path"].apply(lambda p: xy_from_stem(p))  # pyright: ignore[reportAttributeAccessIssue]
        sampled_train_files_time0["__x"] = [xy[0] for xy in xy_pairs]
        sampled_train_files_time0["__y"] = [xy[1] for xy in xy_pairs]
        sampled_train_files_time0 = sampled_train_files_time0.sort_values(by=["__x", "__y"]).drop(  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
            columns=["__x", "__y"]
        )
    elif hasattr(eval_strat, "split") and eval_strat.split is not None:  # pyright: ignore[reportAttributeAccessIssue]
        logger.info(f"Using {eval_strat.split} split to select starting samples")  # pyright: ignore[reportAttributeAccessIssue]
        split_samples = pd.read_parquet(cfg.output_dir.parent / f"{eval_strat.split}_samples.parquet")  # pyright: ignore[reportAttributeAccessIssue]
        sampled_train_files_time0 = all_train_files_discrete_time_0[
            all_train_files_discrete_time_0["file_path"].isin(split_samples["file_path"])  # pyright: ignore[reportAttributeAccessIssue]
        ]
        assert len(sampled_train_files_time0) >= eval_strat.nb_generated_samples, (
            f"Expected at least {eval_strat.nb_generated_samples} samples from {eval_strat.split} split, got {len(sampled_train_files_time0)}"  # pyright: ignore[reportAttributeAccessIssue]
        )
        logger.debug(
            f"Filtered starting samples from {len(df)} to {len(sampled_train_files_time0)} using split {eval_strat.split}"  # pyright: ignore[reportAttributeAccessIssue]
        )
        sampled_train_files_time0 = sampled_train_files_time0.sample(eval_strat.nb_generated_samples)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        sampled_train_files_time0 = all_train_files_discrete_time_0.sample(eval_strat.nb_generated_samples)

    ds_output = starting_ds.__getitems__(sampled_train_files_time0.index.to_list())  # pyright: ignore[reportArgumentType]
    # stack the outputs into tensors
    starting_batch: Tensor = torch.stack([out.tensor for out in ds_output]).to(device, dtype)  # pyright: ignore[reportCallIssue, reportArgumentType]
    starting_times = torch.tensor([out.time for out in ds_output], device=device, dtype=dtype)  # pyright: ignore[reportArgumentType]
    logger.info(
        f"Using starting data of shape {starting_batch.shape} and type {starting_batch.dtype} at times {starting_times}"
    )
    # write the file names to disk
    with open(base_save_path / "starting_batch_file_names.txt", "w") as f:
        for _, sample in sampled_train_files_time0.iterrows():
            f.write(f"{sample['file_path']}\n")
    # return
    return starting_batch, starting_times


def _get_starting_batch_discrete(
    cfg: InferenceConfig,
    eval_strat: InvertedRegeneration
    | SimpleGeneration
    | ForwardNoising
    | InversionRegenerationOnly
    | VideoGenerationFromNoise,
    logger: MultiProcessAdapter,
    dataset_path: Path | str,
    device: torch.device | str,
    dtype: torch.dtype | str,
) -> tuple[Tensor, Literal[0]]:
    assert cfg.dataset.dataset_params is not None

    database_path = Path(cfg.dataset.path)
    logger.info(f"Using dataset {cfg.dataset.name} from {database_path}")
    subdirs: list[Path] = [  # TODO: redundant with main code
        e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")
    ]
    subdirs.sort(key=cfg.dataset.dataset_params.sorting_func)
    logger.info(f"Using subdir '{subdirs[0].name}' as starting point")
    # ensure no transforms here
    starting_samples = list(subdirs[0].glob(f"*.{cfg.dataset.dataset_params.file_extension}"))
    kept_transforms = remove_flips_and_rotations_from_transforms(cfg.dataset.transforms)[0]
    starting_ds: BaseDataset = cfg.dataset.dataset_params.dataset_class(
        starting_samples,
        kept_transforms,
        cfg.dataset.expected_initial_data_range,
    )
    logger.info(f"Built starting dataset from {subdirs[0]}:\n{starting_ds}")
    logger.info(f"Using transforms:\n{kept_transforms}")

    # select starting samples to generate from
    if hasattr(eval_strat, "plate_name_to_simulate") and eval_strat.plate_name_to_simulate is not None:  # pyright: ignore[reportAttributeAccessIssue]
        # if a plate was given, select nb_generated_samples from it, in order
        glob_pattern = (  # TODO: this is hard-coded for biotine
            f"{eval_strat.plate_name_to_simulate}_time_1_patch_*_*.{cfg.dataset.dataset_params.file_extension}"  # pyright: ignore[reportAttributeAccessIssue]
        )
        # assuming they are sorted in (x,y) dictionary order
        all_matching_sample_names = list(Path(dataset_path, "1").glob(glob_pattern))
        assert len(all_matching_sample_names) > 0, f"No samples found with glob pattern {glob_pattern}"
        logger.info(f"Found {len(all_matching_sample_names)} patches from plate {eval_strat.plate_name_to_simulate}")  # pyright: ignore[reportAttributeAccessIssue]
        # select the first nb_generated_samples
        sample_names = all_matching_sample_names[: eval_strat.nb_generated_samples]
        if len(sample_names) != len(all_matching_sample_names):
            logger.warning(f"Selected {len(sample_names)} patches out of {len(all_matching_sample_names)}")
        assert len(sample_names) == eval_strat.nb_generated_samples, (
            f"Expected to get at least {eval_strat.nb_generated_samples} samples, got {len(sample_names)}"
        )
        logger.info(f"Selected {len(sample_names)} samples to run inference from.")
        tensors = starting_ds.get_items_by_name(sample_names)
    else:
        # if not, select nb_generated_samples samples randomly
        sample_idxes: list[int] = (  # pyright: ignore[reportAssignmentType]
            np.random.default_rng().choice(len(starting_ds), eval_strat.nb_generated_samples, replace=False).tolist()
        )
        tensors = starting_ds.__getitems__(sample_idxes)
        logger.info(f"Selected {len(sample_idxes)} samples to run inference from at random (if needed)")

    starting_batch: Tensor = torch.stack(tensors).to(device, dtype)  # pyright: ignore[reportCallIssue, reportArgumentType]
    logger.debug(f"Using starting data of shape {starting_batch.shape} and type {starting_batch.dtype}")

    return starting_batch, 0


def clean_inference_strategy_folder(
    folder: Path,
    logger: MultiProcessAdapter | logging.Logger,
    config_snapshot: str,
    config_filename: str,
    names_to_ignore: list[str] | None = None,
):
    """
    Cleans the output folder for inference strategy by deleting all files and folders in it but `logs.log`.

    Does not handle distribution!
    """
    logger.info(f"Saving outputs to {bold(folder.parts[-3])}/{bold(folder.parts[-2])}/{bold(folder.name)}")

    # clean the folder if it already exists
    if names_to_ignore is None:
        names_to_ignore = []

    existing_problematic_files = [f for f in folder.iterdir() if f.name != "logs.log" and f.name not in names_to_ignore]
    if len(existing_problematic_files) != 0:
        logger.warning(
            f"Output directory {bold(folder.name)} already exists and is not empty:\n\n{[f.name for f in existing_problematic_files]}"
        )
        print()
        inpt = input("\n    => Overwrite these files/folders (will exist otherwise)? (y/[n]) ")
        if inpt.lower() != "y":
            logger.error("Refusing to proceed with a non-cleared folder.")
            sys.exit(1)
        else:
            logger.info(f"Deleting files/folders: {[f.name for f in existing_problematic_files]}")
            for f in existing_problematic_files:
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    f.unlink()

    # write the config snapshot to the folder
    config_dest_path = folder / config_filename
    if config_dest_path.exists():
        raise RuntimeError(f"Config file {config_dest_path} already exists in destination folder!")
    config_dest_path.write_text(config_snapshot, encoding="utf-8")
    logger.info(f"Wrote config snapshot '{config_filename}' to:\n{config_dest_path}")


if __name__ == "__main__":
    # Load the config
    from my_conf import my_inference_conf
    from my_conf.my_inference_conf import inference_conf

    # Snapshot the config file content once at startup
    config_snapshot = Path(my_inference_conf.__file__).read_text(encoding="utf-8")
    config_filename = Path(my_inference_conf.__file__).name

    prof_conf = inference_conf.profiling
    logger = get_distrib_logger(inference_conf)

    if prof_conf.enabled:
        print("Profiling is enabled")

        # trace_handler = tensorboard_trace_handler("pytorch_traces", use_gzip=True) #TODO:gh-pytorch#136040
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=prof_conf.record_shapes,
            profile_memory=prof_conf.profile_memory,
            with_stack=prof_conf.with_stack,
            with_flops=prof_conf.with_flops,
            # on_trace_ready=trace_handler,
        )

        # profile the main function
        profiler.start()
        main(inference_conf, logger, config_snapshot, config_filename)
        profiler.stop()

        # save full profiling trace (very large)
        if prof_conf.export_chrome_trace:
            logger.info("Saving profiling trace to profiling_trace.json...")
            profiler.export_chrome_trace("profiling_trace.json")
            logger.info("Saved profiling trace to profiling_trace.json")

        # torch.save profiling results
        logger.info("Saving top CPU calls...")
        with Path("profiling_results_self_cpu.txt").open("w") as f:
            avgs = profiler.key_averages(group_by_stack_n=5) if profiler.with_stack else profiler.key_averages()
            f.write(avgs.table(sort_by="self_cpu_time_total", row_limit=20))
        logger.info("Saved top CPU calls at profiling_results_self_cpu.txt")

        logger.info("Saving top CUDA calls...")
        with Path("profiling_results_cuda.txt").open("w") as f:
            avgs = profiler.key_averages(group_by_stack_n=5) if profiler.with_stack else profiler.key_averages()
            f.write(avgs.table(sort_by="cuda_time_total", row_limit=20))
        logger.info("Saved top CUDA calls at profiling_results_cuda.txt")

    else:
        main(inference_conf, logger, config_snapshot, config_filename)
