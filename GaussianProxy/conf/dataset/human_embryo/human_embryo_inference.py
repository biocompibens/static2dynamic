from PIL import ImageFile
from torch import float32
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, Resize

from GaussianProxy.conf.training_conf import DataSet, DatasetParams
from GaussianProxy.utils.data import ImageDataset1D

DEFINITION = 256
NUMBER_OF_CHANNELS = 1

transforms = Compose(
    transforms=[
        Resize(DEFINITION),
        ConvertImageDtype(float32),
        Normalize(mean=[0.5] * NUMBER_OF_CHANNELS, std=[0.5] * NUMBER_OF_CHANNELS),
    ]
)

PHASES_ORDER = ("tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+", "tM", "tSB", "tB", "tEB")

ds_params = DatasetParams(
    file_extension="jpeg",
    key_transform=str,
    sorting_func=lambda subdir: PHASES_ORDER.index(subdir.name),
    dataset_class=ImageDataset1D,
)

dataset = DataSet(
    name="human_embryo",
    data_shape=(NUMBER_OF_CHANNELS, DEFINITION, DEFINITION),
    transforms=transforms,
    selected_dists=None,  # not used
    expected_initial_data_range=(0, 255),
    dataset_params=ds_params,
)

# some jpeg files have premature ending
ImageFile.LOAD_TRUNCATED_IMAGES = True
