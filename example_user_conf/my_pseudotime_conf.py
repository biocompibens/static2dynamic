import random
from pathlib import Path

from GaussianProxy.conf.pseudotime_conf import Params

###################################################################################################################
#################################################### Datasets #####################################################
###################################################################################################################
# isort: off
from my_conf.dataset.dataset_conf import dataset

base_save_dir = Path("/path/to/base/save/dir")

# fmt: off
params = Params(
    base_save_dir              = base_save_dir,
    experiment_names           = "experiment_name",
    datasets                   = [dataset],
    device                     = "cuda:0",
    model_name                 = "facebook/dinov2-with-registers-giant",
    batch_size                 = 512,
    use_model_preprocessor     = False,
    recompute_encodings        = "no-overwrite",
    save_policy                = "ask-before-overwrite",
    seed                       = random.randint(0, 2**32 - 1),
    spline_continuation_range  = (0.3, 0.3),
    nb_times_spline_eval       = 50_000,
    test_split_frac            = 0.1,
    concatenate_train_test     = True,
    times_spacing_method       = "evenly_spaced",
    refit_models               = True,
)
# fmt: on
