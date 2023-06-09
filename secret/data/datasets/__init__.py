from __future__ import absolute_import
import warnings

from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .msmt17 import MSMT17
from .custom import CustomDataset

__factory = {
    'msmt17': MSMT17,
    'dukemtmc': DukeMTMC,
    'market1501': Market1501,
    'canifa': CustomDataset,
    'pmc_reid_dyno_raw' : CustomDataset,
    '20220721_images_split_rotate': CustomDataset,
    'PMC_sup_20220411': CustomDataset,
}

def names():
    return sorted(__factory.keys())

def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if 'PMC' in name or 'pmc' in name:
        return CustomDataset(root, *args, **kwargs)
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
