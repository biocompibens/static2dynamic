from torch import float32
from torchvision.transforms.v2 import Compose, Normalize, ToDtype

from GaussianProxy.conf.training_conf import DataSet, DatasetParams
from GaussianProxy.utils.data import ImageDataset

DEFINITION = 256
NUMBER_OF_CHANNELS = 3

transforms = Compose(
    transforms=[
        ToDtype(float32, scale=True),
        Normalize(mean=[0.5] * NUMBER_OF_CHANNELS, std=[0.5] * NUMBER_OF_CHANNELS),
    ]
)

ds_params = DatasetParams(
    file_extension="png",
    key_transform=int,
    sorting_func=lambda subdir: int(subdir.name),
    dataset_class=ImageDataset,
)

dataset = DataSet(
    name="diabetic_retinopathy_full_circle_augs_2048_precrop",
    data_shape=(NUMBER_OF_CHANNELS, DEFINITION, DEFINITION),
    transforms=transforms,
    selected_dists=None,  # not used
    expected_initial_data_range=(0, 255),
    dataset_params=ds_params,
)
