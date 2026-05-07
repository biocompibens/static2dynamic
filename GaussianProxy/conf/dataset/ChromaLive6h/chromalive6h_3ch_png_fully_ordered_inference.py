from dataclasses import replace

from GaussianProxy.conf.dataset.ChromaLive6h.chromalive6h_3ch_png_inference import dataset
from GaussianProxy.utils.data import ContinuousTimeImageDataset

assert dataset.dataset_params is not None
ds_params = replace(dataset.dataset_params, dataset_class=ContinuousTimeImageDataset)

dataset = replace(dataset, fully_ordered=True, dataset_params=ds_params)
