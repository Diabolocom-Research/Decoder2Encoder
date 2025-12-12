from dataclasses import dataclass


@dataclass
class DatasetConfig:
    data_mix_path: str = "./examples/mix"
    shuffle: bool = True
    batch_size: int = 12
    prefetch_factor: int = 1
    num_workers: int = 1
    predownload: int = 1
    length: int = 2048
    var_len: bool = False
    num_canonical_nodes: int = 0
    pin_memory: bool = True
    step_to_skip: int = 0
    seed: int = 42
    add_bos_token: bool = False
    add_eos_token: bool = False
