from dataclasses import replace

from torch import float32
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.v2 import CenterCrop

from GaussianProxy.conf.dataset.diabetic_retinopathy.diabetic_retinopathy_inference import dataset

transforms = Compose(
    transforms=[
        CenterCrop(2048),
        Resize(256),
        ConvertImageDtype(float32),
        Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)

dataset = replace(dataset, transforms=transforms, name="diabetic_retinopathy_2048_crop")
