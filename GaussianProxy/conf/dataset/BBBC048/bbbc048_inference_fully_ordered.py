from torch import float32
from torchvision.transforms.v2 import Compose, Normalize, ToDtype

from GaussianProxy.conf.training_conf import DataSet, DatasetParams
from GaussianProxy.utils.data import ContinuousTimeImageDataset1D

DEFINITION = 48
NUMBER_OF_CHANNELS = 1

transforms = Compose(
    transforms=[
        ToDtype(float32, scale=True),
        Normalize(mean=[0.5] * NUMBER_OF_CHANNELS, std=[0.5] * NUMBER_OF_CHANNELS),
    ]
)

phase_order = (
    "G1",
    "S",
    "G2",
    "Prophase",
    "Metaphase",
    "Anaphase",
    "Telophase",
)
phase_order_dict = {phase: index for index, phase in enumerate(phase_order)}
ds_params = DatasetParams(
    file_extension="png",
    key_transform=str,
    sorting_func=lambda subdir: phase_order_dict[subdir.name],
    dataset_class=ContinuousTimeImageDataset1D,
)

dataset = DataSet(
    name="BBBC048_fully_ordered",
    data_shape=(NUMBER_OF_CHANNELS, DEFINITION, DEFINITION),
    transforms=transforms,
    selected_dists=None,  # not used
    expected_initial_data_range=(0, 255),
    dataset_params=ds_params,
    fully_ordered=True,
)
