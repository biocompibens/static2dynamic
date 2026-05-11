from dataclasses import replace

from GaussianProxy.conf.dataset.NASH_steatosis.NASH_steatosis_inference import dataset
from GaussianProxy.utils.data import ContinuousTimeImageDataset

assert dataset.dataset_params is not None

updated_params = replace(dataset.dataset_params, dataset_class=ContinuousTimeImageDataset)

dataset = replace(dataset, fully_ordered=True, dataset_params=updated_params)
