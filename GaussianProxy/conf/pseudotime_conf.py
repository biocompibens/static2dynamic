from pathlib import Path
from typing import Literal

import attrs

from GaussianProxy.conf.training_conf import DataSet
from GaussianProxy.utils.data import BaseDataset


@attrs.define(kw_only=True)
class Params:
    """
    - `test_split_frac`: if None, no test split is performed, otherwise the test samples sampling is stratified by labels
    - `spline_continuation_range`: time ranges to use for t_min, t_max parametrization of the spline
    beyond the [0;1] time range defined by the extremal class centroids.
    - `spline_bc_type`: the boundary condition for the spline.
    - `test_regex`: a regex for test set; eg: `"non_annotated_samples"` or `r"A_13_fld_2_time_\\d+_patch_.*"`.
    - `concatenate_train_test`: if True, the train and test sets predictions will be concatenated at the end;
    if False, only the test predictions will be saved. Useful when test is actually all samples.
    - `experiment_name`: the name of the experiment; leave empty for default behavior naming.
    - `precomputed_encodings_path`: full path to force loading some preexisting encodings from there
    - `times_spacing_method`: method to use to define the times associated to each class centroid
    - `refit_models`: whether to refit the models (PCA, UMAPs, LDA, spline) or to use pre-fitted ones saved on disk
    - `fitted_models_path`: a parent path from where to load pre-fitted models if `refit_models` is False; defaults to the base output dir if None
    - `filter_ds_regex`: if not None, only samples whose name contains this regex will be used (either for train or test set)
    - `unpaired_time_regexes`: if not None, keep one frame per video by removing this regex-matched
    time part from each sample filename, then balancing the selected number of frames per matched time token
    """

    base_save_dir: Path
    datasets: list[DataSet]
    device: str
    model_name: str
    batch_size: int
    use_model_preprocessor: bool
    recompute_encodings: Literal["force-overwrite", "no-overwrite", "no"]
    save_policy: Literal["no-overwrite", "overwrite", "ask-before-overwrite"]
    seed: int
    spline_continuation_range: tuple[float, float]
    nb_times_spline_eval: int
    times_spacing_method: Literal["evenly_spaced", "centroids_distance"]
    refit_models: bool
    test_split_frac: float | None = None
    dataset_type: type[BaseDataset] | None = None
    spline_bc_type: Literal["natural", "periodic", "clamped"] = "natural"
    test_regexes: list[str] | list[str | None] | str | None = None
    concatenate_train_test: bool = True
    experiment_names: list[str] | str = ""
    precomputed_encodings_path: str | Path | None = None
    filter_ds_regex: list[str] | str | None = None
    unpaired_time_regexes: list[str] | list[str | None] | str | None = None
    fitted_models_path: Path | None = None

    def __attrs_post_init__(self):
        # check incompatible conditions
        if self.spline_bc_type == "periodic" and self.spline_continuation_range != (0, 0):
            raise ValueError(
                f"Cannot use 'periodic' spline boundary condition with non-zero continuation range {self.spline_continuation_range}. "
                "Set `spline_continuation_range` to (0, 0) or use 'natural' boundary condition."
            )
        if self.test_regexes is not None and self.test_split_frac is not None:
            raise ValueError(
                f"Cannot use train/test split fraction (got {self.test_split_frac}) with test regex (got {self.test_regexes})."
            )
        # check list lengths mismatches
        if isinstance(self.test_regexes, list) and len(self.test_regexes) != len(self.datasets):
            raise ValueError(
                f"Length of test_regexes ({len(self.test_regexes)}, {self.test_regexes}) must match length of datasets ({len(self.datasets)})."
            )
        if isinstance(self.experiment_names, list) and len(self.experiment_names) != len(self.datasets):
            raise ValueError(
                f"Length of experiment_names ({len(self.experiment_names)}, {self.experiment_names}) must match length of datasets ({len(self.datasets)})."
            )
        if isinstance(self.filter_ds_regex, list) and len(self.filter_ds_regex) != len(self.datasets):
            raise ValueError(
                f"Length of filter_ds_regex ({len(self.filter_ds_regex)}, {self.filter_ds_regex}) must match length of datasets ({len(self.datasets)})."
            )
        if isinstance(self.unpaired_time_regexes, list) and len(self.unpaired_time_regexes) != len(self.datasets):
            raise ValueError(
                f"Length of unpaired_time_regexes ({len(self.unpaired_time_regexes)}, {self.unpaired_time_regexes}) must match length of datasets ({len(self.datasets)})."
            )
