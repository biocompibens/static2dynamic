# Static2Dynamic: Reconstructing videos of unobservable cellular, developmental, and disease processes

![Demo](media/cell_cycle/demo.gif)
![Demo](media/ependymal/demo.gif)
![Demo](media/nash/demo.gif)
![Demo](media/retino/demo.gif)
![Demo](media/docetaxel/demo.gif)
![Demo](media/nocodazole/demo.gif)

<p align="center">
  <a href="https://biocompibens.github.io/static2dynamic">
    <img src="https://img.shields.io/badge/Website-Static2Dynamic-blue?style=for-the-badge" alt="Website">
  </a>
  <a href="https://www.biorxiv.org/content/10.64898/2026.05.18.725860">
    <img src="https://img.shields.io/badge/bioRxiv-Preprint-B31B1B?style=for-the-badge" alt="bioRxiv preprint">
  </a>
  <a href="https://huggingface.co/thethomasboyer/Static2Dynamic">
    <img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=for-the-badge" alt="Hugging Face models">
  </a>
</p>

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://github.com/biocompibens/static2dynamic/actions/workflows/pyright.yml/badge.svg)](https://github.com/biocompibens/static2dynamic/actions/workflows/pyright.yml)
[![Ruff Lint](https://github.com/biocompibens/static2dynamic/actions/workflows/ruff.yml/badge.svg)](https://github.com/biocompibens/static2dynamic/actions/workflows/ruff.yml)

## Installation

Static2Dynamic runs on Linux (tested on Ubuntu 20.04.1) and is installed with [`uv`](https://docs.astral.sh/uv).  

It will automatically use Python `3.13` (`uv` will prompt to install if, it is not already).

Install the environment with:

```sh
uv sync --frozen
```

This should be quite fast, and includes all needed dependencies.

Note that in order to get reasonable performance, Static2Dynamic necessitates GPUs to run. It was only tested on NVIDIA hardware, and relies on CUDA.

## Configuration of Static2Dynamic

Static2Dynamic has 3 stages: pseudotime estimation, diffusion training, and video inference.

The configuration of each of these stages follows the same idea: a config *class* is (already) defined in `GaussianProxy/conf/{pseudotime_conf,training_conf,inference_conf}.py`.
Users then need to *import* these generic class definitions and *instantiate* their own user/machine-specific config *objects* under `my_conf/{my_pseudotime_conf,my_training_conf,my_inference_conf}.py`.

Examples of such user configs can be found under `example_user_conf/`.

You can use them as starting points by running these commands:

```sh
mkdir my_conf
cp example_user_conf/*.py my_conf/
```

They still need to be specialized to your paths and actual settings.

For *training* Static2Dynamic, keep reading.  
If you are interested in directly performing inference on pretrained models, you can skip the 2 next sections and jump to the [Video inference](#video-inference) part.

### Pseudotime estimation

The first stage (pseudotime estimation) is performed with the `GaussianProxy/pseudotime_estimation.py` script.

First, a user config must be created. Such a config is a `GaussianProx.conf.pseudotime_conf.Params` *object* and must be defined in a python file located at `my_conf/my_pseudotime_conf.py` and named `params`.

Then the pseudotime estimation script can be ran with the following command:

```sh
uv run GaussianProxy/pseudotime_estimation.py
```

This stage only needs a CPU to run.

### Diffusion training

The second stage (diffusion training) is performed with the `GaussianProxy/train.py` script.

First, a user config must be created. Such a config is a `GaussianProxy.conf.training_conf.Config` *object* and must be defined in a python file located at `my_conf/my_training_conf.py` and named `config`.

Then the diffusion training script can be ran with the following command:

```sh
uv run launcher.py run_name=<run_name> project=<project_name>
```

The launcher script takes care of copying the user config to the experiment folder and launching the actual distributed training script. It also supports SLURM clusters.

### Video inference

The third stage (video inference) is performed with the `GaussianProxy/inference.py` script. It can of course be performed on any already trained model.

Already trained models can be downloaded from [huggingface.co/thethomasboyer/Static2Dynamic]([huggingface.co/thethomasboyer/Static2Dynamic](https://huggingface.co/thethomasboyer/Static2Dynamic)).

First, a user config must be created. Such a config is a `GaussianProxy.conf.inference_conf.InferenceConfig` *object* and must be defined in a python file located at `my_conf/my_inference_conf.py` and named `inference_conf`.

An example can be found in `example_user_conf/my_inference_conf.py`.

The key things to change are typically:

- the dataset path (replace `from my_conf.dataset.dataset_conf import dataset` with the actual dataset config).
- the path to the saved model repo (`root_experiments_path/project_name/folder_name`, all of them needing to be set)
- The GPU `device`

Then the video inference script can be ran with the following command:

```sh
uv run GaussianProxy/inference.py
```

A Static2Dynamic generation corresponds to the `InvertedRegeneration` evaluation class, which is already in the example config. It will generate a grid of videos starting on randomly sampled test frames from the starting dataset (typically a few dozen minutes to a few hours, depending on hardware).
