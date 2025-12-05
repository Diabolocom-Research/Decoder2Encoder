import collections
import concurrent.futures
import gc
import itertools
import json
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, TypedDict

import fire
import numpy as np
import streaming
import streaming.base.format.mds
import streaming.base.util
from streaming.base.format.mds import MDSReader

from optimus.dataprocess.packing_utils import PACKING_ALGOS, create_packing_strategy

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")


# Constants
MIN_SEQUENCE_SIZE = 12
LOG_INTERVAL = 10000
MDS_VERSION = 2


# Type definitions
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
    sample_packing_group_size: int | None = 5,
    val_size: int | None = None,
    size_limit: int | str = "64MB",
    head: int | None = None,
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
        block_size: Target size for packed sequences
        packing_algorithm: One of ['greedy', 'first_fit_shuffle', 'first_fit_decreasing']
        sample_packing_group_size: Number of shards to process before running packing optimization.
            Lower values use less memory but may be less efficient. None processes all shards
            together for optimal packing but uses more memory. Default: 5.
        val_size: Total number of records for validation set (distributed across directories)
        size_limit: Maximum shard size (default: '64MB')
        head: Optional limit on records to process per directory
        num_workers: Number of parallel workers for processing directories
        random_size: If True, use random block sizes between 12 and block_size for greedy packing
        seed: Random seed for reproducibility (applied globally before forking workers)
    """
    all_algorithms = ["greedy"] + PACKING_ALGOS
    if packing_algorithm not in all_algorithms:
        raise ValueError(
            f"Invalid packing algorithm '{packing_algorithm}'. Must be one of {all_algorithms}"
        )

    np.random.seed(seed)
    print(f"Packing algorithm: {packing_algorithm}")
    print(f"Number of workers: {num_workers}")

    input_path = Path(input_dir)
    directories = sorted([d.name for d in input_path.iterdir() if d.is_dir()]) or [""]

    val_sizes = _distribute_val_size(directories, val_size)

    jobs_args = []
    for d, val in zip(directories, val_sizes):
        local_dir = input_path / d if d else input_path
        train_dir = Path(output_dir) / "train" / d if d else Path(output_dir) / "train"
        val_dir = Path(output_dir) / "val" / d if d and val else (Path(output_dir) / "val" if val else None)

        jobs_args.append({
            "local_dir": str(local_dir),
            "train_dir": str(train_dir),
            "val_dir": str(val_dir) if val_dir else None,
            "block_size": block_size,
            "packing_algorithm": packing_algorithm,
            "sample_packing_group_size": sample_packing_group_size,
            "val_size": val,
            "size_limit": size_limit,
            "head": head,
            "random_size": random_size,
        })

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker, args) for args in jobs_args]
        for future in futures:
            future.result()
    print("All workers finished.")

    streaming.base.util.merge_index(str(Path(output_dir) / "train"), keep_local=True)
    if val_size is not None:
        streaming.base.util.merge_index(str(Path(output_dir) / "val"), keep_local=True)

    metadata = {
        "input_dir": input_dir,
        "block_size": block_size,
        "packing_algorithm": packing_algorithm,
        "val_size": val_size,
        "seed": seed,
        "num_workers": num_workers,
        "directories": directories,
    }
    with open(Path(output_dir) / "packing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"\nPacking completed in {elapsed}")
    print(f"Output saved to {output_dir}")


# Packing Worker


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

    directory = Path(local_dir).name or "root"
    print(f"Directory {directory}: starting...", flush=True)

    start = time.time()
    index = _load_index(local_dir)
    readers = (
        streaming.base.format.mds.MDSReader.from_json(local_dir, None, s)
        for s in index["shards"]
    )

    records = _create_packed_records(
        readers, packing_algorithm, block_size, random_size,
        sample_packing_group_size, head, directory
    )

    val, train = _split_records(records, val_size)

    if val is not None and val_dir:
        print(f"Directory {directory}: writing validation set to {val_dir}", flush=True)
        _write_iterable(val_dir, val, size_limit, f"(Dir {directory}) val")

    print(f"Directory {directory}: writing training set to {train_dir}", flush=True)
    _write_iterable(train_dir, train, size_limit, f"Directory {directory}:")

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"\n{'=' * 40}\nDirectory {directory}: finished in {elapsed}\n{'=' * 40}\n", flush=True)


def _create_packed_records(
    readers: Iterable[MDSReader],
    packing_algorithm: str,
    block_size: int,
    random_size: bool,
    sample_packing_group_size: int | None,
    head: int | None,
    directory: str,
) -> Iterable[dict[str, Any]]:
    if packing_algorithm == "greedy":
        records = itertools.chain.from_iterable(readers)
        if head is not None:
            print(f"Directory {directory}: taking only the first {head} records.", flush=True)
            records = itertools.islice(records, head)
        records = map(_from_numpy, records)
        records = _pack_greedy(records, block_size, random_size)
    else:
        packed_data = _pack_optimized(
            readers,
            block_size=block_size,
            packing_algorithm=packing_algorithm,
            sample_packing_group_size=sample_packing_group_size,
            head=head,
            desc_prefix=f"Directory {directory}: ",
        )
        records = packed_data

    return map(_to_numpy, records)


def _pack_greedy(
    records: Iterable[TokRecord],
    block_size: int,
    random_size: bool = False,
) -> Iterable[PackTokRecord]:
    def _empty() -> PackTokRecord:
        return {"tokens": [], "metadata": [], "cu_seqlens": [0]}

    def _append(curr: PackTokRecord, doc: TokRecord, size: int) -> tuple[PackTokRecord, TokRecord]:
        leftover_size = size - len(curr["tokens"])
        add_size = min(leftover_size, len(doc["tokens"]))

        curr["tokens"].extend(doc["tokens"][:add_size])
        doc["tokens"] = doc["tokens"][add_size:]

        curr["cu_seqlens"].append(len(curr["tokens"]))
        curr["metadata"].append(doc["metadata"])
        return curr, doc

    curr = _empty()
    current_size = np.random.randint(MIN_SEQUENCE_SIZE, block_size + 1) if random_size else block_size

    for doc in records:
        while len(doc["tokens"]) > 0:
            curr, doc = _append(curr, doc, size=current_size)

            if len(curr["tokens"]) == current_size:
                if random_size:
                    curr["cu_seqlens"] = [0, current_size]
                    current_size = np.random.randint(MIN_SEQUENCE_SIZE, block_size + 1)
                yield curr
                curr = _empty()


def _pack_optimized(
    readers: Iterable[MDSReader],
    block_size: int,
    packing_algorithm: str,
    sample_packing_group_size: int | None = None,
    head: int | None = None,
    desc_prefix: str = "",
) -> Iterable[PackTokRecord]:
    packing_metadata = []

    for histogram, sequence_locations, readers_dict in _build_histogram_streaming(
        readers=readers,
        block_size=block_size,
        sample_packing_group_size=sample_packing_group_size,
        head=head,
    ):
        assignments, metadata = create_packing_strategy(histogram, block_size, packing_algorithm)
        packing_metadata.append(metadata)

        packed_data = _fill_packing_strategy(
            assignments, sequence_locations, readers_dict, block_size
        )

        yield from packed_data

    _print_packing_stats(packing_metadata, desc_prefix=desc_prefix)


# Efficient packing tools


def _build_histogram_streaming(
    readers: Iterable[MDSReader],
    block_size: int,
    sample_packing_group_size: int | None = None,
    head: int | None = None,
) -> Iterable[tuple[list[int], dict[int, list[tuple[int, int]]], dict[int, MDSReader]]]:
    histogram = [0] * (block_size + 1)
    sequence_locations = collections.defaultdict(list)
    readers_dict = {}
    total_records = 0
    sample_packing_count = 0

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


def _fill_packing_strategy(
    assignments: list[list[int]],
    sequence_locations: dict[int, list[tuple[int, int]]],
    readers_dict: dict[int, MDSReader],
    block_size: int,
) -> Iterable[PackTokRecord]:
    for locations in sequence_locations.values():
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


# Tools


def _from_numpy(record: dict[str, Any]) -> TokRecord:
    return {
        "tokens": record["tokens"].tolist(),
        "metadata": record["metadata"],
    }


def _to_numpy(record: TokRecord | PackTokRecord) -> dict[str, Any]:
    return {
        "tokens": np.array(record["tokens"], dtype=np.int32),
        "cu_seqlens": record["cu_seqlens"],
        "metadata": record["metadata"],
    }


def _print_packing_stats(packing_metadata: list[dict[str, Any]], desc_prefix: str = ""):
    if not packing_metadata:
        return

    packing_metadata_agg = collections.defaultdict(list)
    for metadata in packing_metadata:
        for key, value in metadata.items():
            packing_metadata_agg[key].append(value)

    stats = {
        "Dataset max seq length": max(packing_metadata_agg["dataset_max_seqlen"]),
        "Max samples per bin": max(packing_metadata_agg["max_samples_per_bin"]),
        "Packing factor": f"{sum(packing_metadata_agg['packing_factor']) / len(packing_metadata_agg['packing_factor']):.2f}x",
        "Packing efficiency": f"{sum(packing_metadata_agg['packing_efficiency']) / len(packing_metadata_agg['packing_efficiency']):.2f}%",
        "Min packed seq length": min(packing_metadata_agg["min_packed_seqlen"]),
    }

    print(f"{desc_prefix}packing statistics:")
    for key, value in stats.items():
        print(f"{key:25} {value}", flush=True)


def _distribute_val_size(directories: list[str], val_size: int | None) -> list[int | None]:
    if val_size is None:
        return [None] * len(directories)

    val_size_per_dir = val_size // len(directories)
    val_sizes = [val_size_per_dir] * len(directories)

    remainder = val_size % len(directories)
    for i in range(remainder):
        val_sizes[i] += 1

    return val_sizes


def _split_records(
    records: Iterable[TokRecord], val_size: int | None
) -> tuple[Iterable[TokRecord] | None, Iterable[TokRecord]]:
    if val_size is not None:
        val_iter, train_iter = itertools.tee(records, 2)
        val = itertools.islice(val_iter, val_size)
        train = itertools.islice(train_iter, val_size, None)
    else:
        val = None
        train = records

    return val, train


def _load_index(local_dir: str) -> dict[str, Any]:
    index_path = Path(local_dir) / "index.json"
    with open(index_path, "r") as f:
        index = json.load(f)

    if index["version"] != MDS_VERSION:
        raise ValueError(f"Invalid index version: {index['version']}, expected {MDS_VERSION}.")

    return index


def _write_iterable(
    output_dir: str,
    records: Iterable[dict[str, Any]],
    size_limit: int | str,
    desc_prefix: str = "",
):
    columns = {
        "tokens": "ndarray:int32",
        "metadata": "json",
        "cu_seqlens": "json",
    }
    start = time.time()

    def _format_progress(count: int, elapsed: float) -> str:
        records_per_second = int(count / elapsed) if elapsed > 0 else 0
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        return f"{desc_prefix} processed {count} records in {hours}H:{minutes}m:{seconds:.2f}s ({records_per_second} Records/s)."

    with streaming.MDSWriter(out=output_dir, columns=columns, size_limit=size_limit) as out:
        for i, record in enumerate(records, start=1):
            out.write(record)
            if i % LOG_INTERVAL == 0:
                print(_format_progress(i, time.time() - start), flush=True)
                gc.collect()
        print(_format_progress(i, time.time() - start), flush=True)


# CLI


if __name__ == "__main__":
    fire.Fire(pack_dataset)
