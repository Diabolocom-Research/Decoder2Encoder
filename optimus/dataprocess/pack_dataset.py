import collections
import concurrent.futures
import gc
import itertools
import json
import pathlib
import time
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

import fire
import numpy as np
import streaming
import streaming.base.format.mds
import streaming.base.util
from streaming.base.format.mds import MDSReader

from optimus.dataprocess.packing_utils import (
    PACKING_ALGOS,
    create_packing_strategy,
)

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*found in sys.modules.*"
)


Metadata = dict[str, Any] | list[Any]


class TokRecord(TypedDict):
    tokens: list[int]
    metadata: Any


class PackTokRecord(TypedDict):
    tokens: list[int]
    metadata: list[Any]
    cu_seqlens: list[int]


def pack_dataset(
    input_dir: str,
    output_dir: str,
    block_size: int,
    packing_algorithm: str = "greedy",
    sample_packing_group_size: Optional[int] = 5,
    val_size: Optional[int] = None,
    size_limit: Optional[int | str] = "64MB",
    head: Optional[int] = None,
    num_workers: int = 1,
    random_size: bool = False,
    seed: int = 42,
):
    """
    Pack a tokenized dataset into fixed-size blocks.

    Supports multiple packing algorithms:
    - 'greedy': Fast streaming concatenation
    - 'first_fit_shuffle': Optimized bin packing with shuffling
    - 'first_fit_decreasing': Optimized bin packing sorted by size

    Args:
        input_dir: Path to tokenized dataset (can contain subdirectories)
        output_dir: Path to save packed dataset
        block_size: Target size for packed sequences (required if packing)
        packing_algorithm: One of ['greedy', 'first_fit_shuffle', 'first_fit_decreasing']
        sample_packing_group_size: Number of samples to group for packing optimization, None for entire dataset (default: 5)
        val_size: Number of records for validation set
        size_limit: Maximum shard size (default: '64MB')
        head: Optional limit on records per directory
        num_workers: Number of parallel workers
        random_size: Use random sizes for greedy packing
        seed: Random seed for reproducibility
    """
    all_algorithms = ["greedy"] + PACKING_ALGOS
    if packing_algorithm not in all_algorithms:
        raise ValueError(
            f"Invalid packing algorithm '{packing_algorithm}'. "
            f"Must be one of {all_algorithms}"
        )

    np.random.seed(seed)
    print(f"Packing algorithm: {packing_algorithm}")
    print(f"Number of workers: {num_workers}")
    directories = [d.name for d in pathlib.Path(input_dir).iterdir() if d.is_dir()]
    directories = sorted(directories) or [""]

    vals = _get_val_sizes(directories, val_size)
    jobs_args = [
        {
            "local_dir": f"{input_dir}/{d}" if d else input_dir,
            "train_dir": f"{output_dir}/train/{d}" if d else f"{output_dir}/train",
            "val_dir": f"{output_dir}/val/{d}" if d and val else None,
            "block_size": block_size,
            "packing_algorithm": packing_algorithm,
            "sample_packing_group_size": sample_packing_group_size,
            "val_size": val,
            "size_limit": size_limit,
            "head": head,
            "random_size": random_size,
        }
        for d, val in zip(directories, vals)
    ]

    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker, args) for args in jobs_args]
        for future in futures:
            future.result()
    print("All workers finished.")

    streaming.base.util.merge_index(f"{output_dir}/train", keep_local=True)
    if val_size is not None:
        streaming.base.util.merge_index(f"{output_dir}/val", keep_local=True)

    metadata = {
        "input_dir": input_dir,
        "block_size": block_size,
        "packing_algorithm": packing_algorithm,
        "val_size": val_size,
        "seed": seed,
        "num_workers": num_workers,
        "directories": directories,
    }
    metadata_path = pathlib.Path(output_dir) / "packing_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    fmt_elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"\nPacking completed in {fmt_elapsed}")
    print(f"Output saved to {output_dir}")


# ================
# Packing Worker
# ================


def _worker(args: dict[str, Any]):
    local_dir = args["local_dir"]
    train_dir = args["train_dir"]
    val_dir = args.get("val_dir")
    block_size = args["block_size"]
    packing_algorithm = args["packing_algorithm"]
    sample_packing_group_size = args["sample_packing_group_size"]
    val_size = args.get("val_size")
    size_limit = args["size_limit"]
    random_size = args.get("random_size", False)
    head = args.get("head")

    directory = pathlib.Path(local_dir).name or "root"
    print(f"Directory {directory}: starting...", flush=True)

    start = time.time()
    index = _load_index(local_dir)
    shards = index["shards"]
    readers = (
        streaming.base.format.mds.MDSReader.from_json(local_dir, None, s)
        for s in shards
    )

    if packing_algorithm == "greedy":
        records = itertools.chain.from_iterable(readers)
        if head is not None:
            print(
                f"Directory {directory}: taking only the first {head} records.",
                flush=True,
            )
            records = itertools.islice(records, head)
        records = map(_from_numpy, records)
        records = _pack(records, block_size, random_size)
        records = map(_to_numpy, records)
    else:
        packed_data = _pack_optimized(
            readers,
            block_size=block_size,
            packing_algorithm=packing_algorithm,
            sample_packing_group_size=sample_packing_group_size,
            head=head,
            desc_prefix=f"Directory {directory}: ",
        )
        records = map(_to_numpy, packed_data)

    val, train = _split_records(records, val_size)
    if val is not None and val_dir:
        print(f"Directory {directory}: writing validation set to {val_dir}", flush=True)
        _write_iterable(val_dir, val, size_limit, f"(Dir {directory}) val")

    print(f"Directory {directory}: writing training set to {train_dir}", flush=True)
    _write_iterable(train_dir, train, size_limit, f"Directory {directory}:")

    elapsed = time.time() - start
    fmt_elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(
        f"\n{'=' * 40}\nDirectory {directory}: finished in {fmt_elapsed}\n{'=' * 40}\n",
        flush=True,
    )


def _pack(
    records: Iterable[TokRecord],
    block_size: int,
    random_size: bool = False,
) -> Iterable[PackTokRecord]:
    def _empty() -> PackTokRecord:
        return {"tokens": [], "metadata": [], "cu_seqlens": [0]}

    def _append(
        curr: PackTokRecord, doc: TokRecord, size: int
    ) -> tuple[PackTokRecord, TokRecord]:
        leftover_size = size - len(curr["tokens"])
        doc_size = len(doc["tokens"])

        add_size = min(leftover_size, doc_size)

        tokens_to_add = doc["tokens"][:add_size]
        curr["tokens"].extend(tokens_to_add)
        doc["tokens"] = doc["tokens"][add_size:]

        curr["cu_seqlens"].append(len(curr["tokens"]))
        curr["metadata"].append(doc["metadata"])
        return curr, doc

    curr = _empty()
    min_doc_size = 12
    current_random_size = np.random.randint(min_doc_size, block_size + 1)

    for doc in records:
        while len(doc["tokens"]) > 0:
            if random_size:
                curr, doc = _append(curr, doc, size=current_random_size)
                if len(curr["tokens"]) == current_random_size:
                    curr["cu_seqlens"] = [0, current_random_size]
                    yield curr
                    curr = _empty()
                    current_random_size = np.random.randint(
                        min_doc_size, block_size + 1
                    )
            else:
                curr, doc = _append(curr, doc, size=block_size)
                if len(curr["tokens"]) == block_size:
                    yield curr
                    curr = _empty()


def _pack_optimized(
    readers: Iterable[MDSReader],
    block_size: int,
    packing_algorithm: str,
    sample_packing_group_size: Optional[int] = None,
    head: Optional[int] = None,
    desc_prefix: str = "",
) -> Iterable[PackTokRecord]:
    """Pack a dataset split using two-pass streaming approach."""
    packing_metadata = []

    for histogram, sequence_locations, readers_dict in _build_histogram_streaming(
        readers=readers,
        block_size=block_size,
        sample_packing_group_size=sample_packing_group_size,
        head=head,
    ):
        assignments, metadata = create_packing_strategy(
            histogram,
            block_size,
            packing_algorithm,
        )
        packing_metadata.append(metadata)

        packed_data = _fill_packing_strategy_streaming(
            assignments,
            sequence_locations,
            readers_dict,
            block_size,
        )

        yield from packed_data
    _print_packing_stats(packing_metadata, desc_prefix=desc_prefix)


# =======================
# Efficient packing tools
# =======================


def _build_histogram_streaming(
    readers: Iterable["MDSReader"],
    block_size: int,
    sample_packing_group_size: Optional[int] = None,
    head: Optional[int] = None,
) -> Iterable[
    Tuple[List[int], Dict[int, List[Tuple[int, int]]], Dict[int, "MDSReader"]]
]:
    """
    Build histograms and sequence locations across readers in a streaming fashion.
    """
    histogram = [0] * (block_size + 1)
    sequence_locations = collections.defaultdict(list)
    readers_dict = {}
    total_records, sample_packing_count = 0, 0

    for shard_idx, reader in enumerate(readers):
        readers_dict[shard_idx] = reader
        for record_idx in range(reader.size):
            if head is not None and total_records >= head:
                yield histogram, sequence_locations, readers_dict
                return

            record = reader.get_item(record_idx)
            seq_len = min(len(record["tokens"]), block_size)

            histogram[seq_len] += 1
            sequence_locations[seq_len].append((shard_idx, record_idx))
            total_records += 1

        if sample_packing_group_size is not None:
            sample_packing_count += 1
            if sample_packing_count >= sample_packing_group_size:
                yield histogram, sequence_locations, readers_dict
                histogram = [0] * (block_size + 1)
                sequence_locations = collections.defaultdict(list)
                readers_dict = {}
                sample_packing_count = 0

    if sequence_locations:
        yield histogram, sequence_locations, readers_dict


def _fill_packing_strategy_streaming(
    assignments: List[List[int]],
    sequence_locations: Dict[int, List[Tuple[int, int]]],
    readers_dict: Dict[int, MDSReader],
    block_size: int,
) -> Iterable[PackTokRecord]:
    """
    Fill bins with actual sequences using random access.
    """
    for _, locations in sequence_locations.items():
        np.random.shuffle(locations)

    for assignment in assignments:
        tokens_buffer = []
        metadata_list = []
        cu_seqlens = [0]

        for seq_length in assignment:
            shard_idx, record_idx = sequence_locations[seq_length].pop()
            record = readers_dict[shard_idx].get_item(record_idx)

            tokens = record["tokens"][:block_size]
            tokens_buffer.extend(tokens.tolist())

            metadata_list.append(record.get("metadata", {}))
            cu_seqlens.append(len(tokens_buffer))

        yield {
            "tokens": tokens_buffer,
            "metadata": metadata_list,
            "cu_seqlens": cu_seqlens,
        }


# ================
# Tools
# ================


def _from_numpy(record: dict[str, Any]) -> TokRecord:
    return {
        "tokens": record["tokens"].tolist(),
        "metadata": record["metadata"],
    }


def _to_numpy(record: TokRecord | PackTokRecord) -> dict[str, Any]:
    result = {
        "tokens": np.array(record["tokens"], dtype=np.int32),
        "cu_seqlens": record["cu_seqlens"],
        "metadata": record["metadata"],
    }
    return result


def _print_packing_stats(packing_metadata: dict[str, Any], desc_prefix: str = ""):
    packing_metadata_agg = collections.defaultdict(list)
    for d in packing_metadata:
        for k, v in d.items():
            packing_metadata_agg[k].append(v)

    stats = {
        "Dataset max seq length": max(packing_metadata_agg["dataset_max_seqlen"]),
        "Max samples per bin": max(packing_metadata_agg["max_samples_per_bin"]),
        "Packing factor": f"{(sum(packing_metadata_agg['packing_factor']) / len(packing_metadata_agg['packing_factor'])):.2f}x",
        "Packing efficiency": f"{(sum(packing_metadata_agg['packing_efficiency']) / len(packing_metadata_agg['packing_efficiency'])):.2f}%",
        "Min packed seq length": min(packing_metadata_agg["min_packed_seqlen"]),
    }

    print(f"{desc_prefix}packing statistics:")
    for k, v in stats.items():
        print(f"{k:25} {v}", flush=True)


def _get_val_sizes(
    directories: list[str], val_size: Optional[int]
) -> list[Optional[int]]:
    # Quantity of validation samples per directory
    if val_size is None:
        return [None] * len(directories)

    val_size_per_dir = val_size // len(directories)
    vals = [val_size_per_dir] * len(directories)
    remainder = val_size % len(directories)
    for i in range(remainder):
        vals[i] += 1
    return vals


def _split_records(
    records: Iterable[TokRecord], val_size: Optional[int]
) -> tuple[Optional[Iterable[TokRecord]], Iterable[TokRecord]]:
    if val_size is not None:
        val = itertools.islice(records, val_size)
        train = records
    else:
        val = None
        train = records

    return val, train


def _load_index(local_dir: str) -> dict[str, Any]:
    index_path = pathlib.Path(local_dir) / "index.json"
    with open(index_path, "r") as f:
        index = json.load(f)

    if index["version"] != 2:
        raise ValueError(f"Invalid index version: {index['version']}, expected 2.")

    return index


def _write_iterable(
    output_dir: str,
    records: Iterable[dict[str, Any]],
    size_limit: Optional[int | str],
    desc_prefix: Optional[str] = None,
):
    columns = {
        "tokens": "ndarray:int32",
        "metadata": "json",
        "cu_seqlens": "json",
    }
    start = time.time()
    prefix = f"{desc_prefix}" if desc_prefix else ""

    def print_status(i: int, start: float):
        ellapsed = time.time() - start
        total_seconds = ellapsed
        records_per_second = int(i / total_seconds)
        msg = (
            f"{prefix} processed {i} records in "
            f"{int(ellapsed // 3600)}H:{int((ellapsed % 3600) // 60)}m:{ellapsed % 60:.2f}s ({records_per_second} Records/s)."
        )
        print(msg, flush=True)

    with streaming.MDSWriter(
        out=output_dir,
        columns=columns,
        size_limit=size_limit,
    ) as out:
        for i, record in enumerate(records, start=1):
            out.write(record)
            if i % 10000 == 0:
                print_status(i, start)
                gc.collect()
        print_status(i, start)


# ================
# CLI
# ================


if __name__ == "__main__":
    fire.Fire(pack_dataset)
