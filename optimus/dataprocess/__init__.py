from optimus.dataprocess.tokenize_dataset import tokenize_dataset
from optimus.dataprocess.subsample_dataset import subsample_dataset
from optimus.dataprocess.inspect_dataset import inspect_dataset


def __getattr__(name):
    if name == "pack_dataset":
        from .pack_dataset import pack_dataset
        return pack_dataset
    raise AttributeError(f"module 'optimus.dataprocess' has no attribute '{name}'")
