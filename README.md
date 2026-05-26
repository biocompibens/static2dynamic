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

It will automatically use Python `3.13` (`uv` will install if it is not already).

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

For *training* Static2Dynamic, keep reading the [Pseudotime estimation](#pseudotime-estimation) and [Diffusion training](#diffusion-training) sections below.

If you are interested in directly performing inference on pretrained models, you can skip the two next sections and jump to the [Video inference](#video-inference) part –see also the [Quick run](#quick-run) section for a detailed example of how to do that.

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

The third stage (video inference) is performed with the `GaussianProxy/inference.py` script.

It can of course be performed on any already trained model; they can be downloaded from [huggingface.co/thethomasboyer/Static2Dynamic]([huggingface.co/thethomasboyer/Static2Dynamic](https://huggingface.co/thethomasboyer/Static2Dynamic)), together with the pseudotime predictions. If reusing precomputed pseudotimes predictions, make sure to update the paths saved in the `.parquet` files to match the actual paths on your machine. A script is provided at `scripts/change_dataset_prefix_in_parquet_file.py` to do this.

First, a user config must be created. Such a config is a `GaussianProxy.conf.inference_conf.InferenceConfig` *object* and must be defined in a python file located at `my_conf/my_inference_conf.py` and named `inference_conf`.

An example can be found in `example_user_conf/my_inference_conf.py`.

The key things to change are typically:

- the dataset path (replace `from my_conf.dataset.dataset_conf import dataset` with the actual dataset config to use)
- the path to the saved model repo (`root_experiments_path/project_name/folder_name`, all of them needing to be set)
- The GPU `device`

Then the video inference script can be ran with the following command:

```sh
uv run GaussianProxy/inference.py
```

A Static2Dynamic generation corresponds to the `InvertedRegeneration` evaluation class, which is already in the example config. It will generate a grid of videos starting on randomly sampled test frames from the starting dataset (typically a few dozen minutes to a few hours, depending on hardware).

## Quick run

Here, as an example, we provide detailed instructions to quickly perform inference on the NASH steatosis dataset using the pretrained model, once the environment [Installation](#installation) is complete.

If the `.envrc` file is not automatically loaded by `direnv`, run it manually to activate the environment and set the `PYTHONPATH` properly.

### Download the data

Download the steatosis data ([Heinemann et al., 2019](https://rdcu.be/fkCxH)) from [osf.io/p48rd](https://osf.io/p48rd) and extract it.

Static2Dynamic is a data-to-data method and thus needs a starting datapoint to perform inference. It will use its own test split if configured to do so (the default). For the steatosis dataset specifically, merge the original splits:

```sh
for d in steatosis/val/*; do
  cls="$(basename "$d")"
  mv "$d"/* "steatosis/training/$cls"/
done
```

### Download the model and pseudotime predictions for the dataset

Download the `NASH_steato` model from [huggingface.co/thethomasboyer/Static2Dynamic](https://huggingface.co/thethomasboyer/Static2Dynamic). Example:

```sh
hf download thethomasboyer/Static2Dynamic --include 'NASH_steato/*' --local-dir ./Static2Dynamic_models
```

This will create a `Static2Dynamic_models/NASH_steato` folder containing the trained model and the pseudotime predictions for the dataset.

Then update the file paths saved in `.Static2Dynamic_models/NASH_steato/pseudotime_predictions/NASH_steatosis__continuous_time_predictions__facebook_dinov2-with-registers-giant_dataset_preproc.parquet` with the `scripts/change_dataset_prefix_in_parquet_file.py` script to match the actual paths on your machine. Example if you extracted the data under `./steatosis/`:

```sh
python scripts/change_dataset_prefix_in_parquet_file.py \
--dataset_files_list Static2Dynamic_models/NASH_steato/pseudotime_predictions/NASH_steatosis__continuous_time_predictions__facebook_dinov2-with-registers-giant_dataset_preproc.parquet \
--previous_dataset_prefix /projects/static2dynamic/datasets/NASH/prepared_data/steatosis \
--new_dataset_prefix ./steatosis/training
```

Do the same for the train/test splits files:

```sh
python scripts/change_dataset_prefix_in_parquet_file.py \
--dataset_files_list Static2Dynamic_models/NASH_steato/{train,test}_samples.parquet \
--previous_dataset_prefix /projects/static2dynamic/datasets/NASH/prepared_data/steatosis \
--new_dataset_prefix ./steatosis/training
```

### Instantiate a user config for inference

Now create a user config for inference. Start by copying the proposed template:

```sh
mkdir my_conf
cp example_user_conf/my_inference_conf.py my_conf/
```

Then create a dataset config in `my_conf/dataset/NASH_steato.py`:

```python
from dataclasses import replace

from GaussianProxy.conf.dataset.NASH_steatosis.NASH_steatosis_fully_ordered_inference import dataset

dataset = replace(
    dataset,
    path = "/path/to/the/extracted/NASH_steatosis/data",
    path_to_single_parquet = "./Static2Dynamic_models/NASH_steato/pseudotime_predictions/NASH_steatosis__continuous_time_predictions__facebook_dinov2-with-registers-giant_dataset_preproc.parquet"
)
```

Replace "`/path/to/the/extracted/NASH_steatosis/data`" with the actual path to the extracted data.

Then replace the line "`from my_conf.dataset.dataset_conf import dataset`" in `my_conf/my_inference_conf.py` with this dataset config: `from my_conf.dataset.NASH_steato import dataset`.

Similarly, set:

```python
root_experiments_path = "."
project_name          = "Static2Dynamic_models"
folder_name           = "NASH_steato"
```

as well as the `device` and any other option you want to change (`compile`, `InvertedRegeneration` options, etc) from the proposed default in `my_conf/my_inference_conf.py`.

### Run inference

Finally, run the inference script:

```sh
uv run GaussianProxy/inference.py
```

It will output under `run_path = root_experiments_path / project_name / folder_name`.
