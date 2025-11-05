import collections
import concurrent.futures
import gc
import itertools
import json
import pathlib
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

import fire
import numpy as np
import streaming
import streaming.base.format.mds
import streaming.base.util

from optimus.dataprocess.packing_utils import (
    create_packing_strategy,
    PACKING_ALGOS,
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
    block_size: Optional[int] = None,
    packing_algorithm: str = "greedy",
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
        val_size: Number of records for validation set
        size_limit: Maximum shard size (default: '64MB')
        head: Optional limit on records per directory
        num_workers: Number of parallel workers
        random_size: Use random sizes for greedy packing
        seed: Random seed for reproducibility
    """
    # Validate packing algorithm
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
            "val_size": val,
            "size_limit": size_limit,
            "head": head,
            "random_size": random_size,
        }
        for d, val in zip(directories, vals)
    ]

    print("\nJobs:")
    for arg in jobs_args:
        print(json.dumps({k: v for k, v in arg.items() if v is not None}, indent=2))

    start_time = time.time()
    print(f"\nStarting {num_workers} workers...")
    
    # Execute jobs
    if num_workers == 1:
        # Single-threaded execution
        for args in jobs_args:
            _worker(args)
    else:
        # Multi-process execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker, args) for args in jobs_args]
            for future in futures:
                future.result()
    
    print("\nAll workers finished.")
    # Merge indexes
    streaming.base.util.merge_index(f"{output_dir}/train", keep_local=True)
    if val_size is not None:
        streaming.base.util.merge_index(f"{output_dir}/val", keep_local=True)
    
    # Save metadata
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


def _get_val_sizes(
    directories: list[str], val_size: Optional[int]
) -> list[Optional[int]]:
    if val_size is None:
        return [None] * len(directories)

    val_size_per_dir = val_size // len(directories)
    vals = [val_size_per_dir] * len(directories)
    remainder = val_size % len(directories)

    for i in range(remainder):
        vals[i] += 1

    return vals


def _worker(args: dict[str, Any]):
    local_dir = args["local_dir"]
    train_dir = args["train_dir"]
    val_dir = args.get("val_dir")
    block_size = args["block_size"]
    packing_algorithm = args["packing_algorithm"]
    val_size = args.get("val_size")
    size_limit = args["size_limit"]
    random_size = args.get("random_size", False)
    head = args.get("head")

    directory = pathlib.Path(local_dir).name or "root"
    
    start = time.time()
    print(f"(Dir {directory}) Starting...", flush=True)
    
    # Branch based on packing algorithm
    if packing_algorithm == "greedy":
        _worker_greedy(
            local_dir, train_dir, val_dir, block_size, val_size,
            size_limit, random_size, head, directory
        )
    else:
        _worker_optimized(
            local_dir, train_dir, val_dir, block_size, packing_algorithm,
            val_size, size_limit, head, directory
        )
    
    elapsed = time.time() - start
    fmt_elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"(Dir {directory}) Finished in {fmt_elapsed}", flush=True)


def _worker_greedy(
    local_dir: str,
    train_dir: str,
    val_dir: Optional[str],
    block_size: int,
    val_size: Optional[int],
    size_limit: str,
    random_size: bool,
    head: Optional[int],
    directory: str,
):
    """Worker using greedy streaming packing."""
    index = _load_index(local_dir)
    shards = index["shards"]

    readers = (
        streaming.base.format.mds.MDSReader.from_json(local_dir, None, s)
        for s in shards
    )
    records = itertools.chain.from_iterable(readers)

    if head is not None:
        print(f"(Dir {directory}) Taking only the first {head} records.", flush=True)
        records = itertools.islice(records, head)

    records = map(_from_numpy, records)

    if block_size is not None:
        records = _pack(records, block_size, random_size, f"{directory} pack")

    records = map(_to_numpy, records)

    val, train = _split_records(records, val_size)

    if val is not None and val_dir:
        print(f"(Dir {directory}) Writing validation set to {val_dir}", flush=True)
        _write_iterable(val_dir, val, size_limit, f"(Dir {directory}) val")

    print(f"(Dir {directory}) Writing training set to {train_dir}", flush=True)
    _write_iterable(train_dir, train, size_limit, f"(Dir {directory}) train")


def _worker_optimized(
    local_dir: str,
    train_dir: str,
    val_dir: Optional[str],
    block_size: int,
    packing_algorithm: str,
    val_size: Optional[int],
    size_limit: str,
    head: Optional[int],
    directory: str,
):
    """Worker using optimized streaming packing."""
    # Pack the dataset using two-pass streaming
    packed_data = _pack_split_streaming(
        local_dir,
        block_size=block_size,
        packing_algorithm=packing_algorithm,
        head=head,
        directory=directory,
    )
    
    # Split train/val if needed
    if val_size is not None and val_size > 0:
        val_packed = packed_data[:val_size]
        train_packed = packed_data[val_size:]
        print(f"(Dir {directory}) Split: {len(train_packed)} train, {len(val_packed)} val", flush=True)
    else:
        train_packed = packed_data
        val_packed = None
    
    # Write outputs
    print(f"(Dir {directory}) Writing training set to {train_dir}", flush=True)
    _write_packed_dataset(train_dir, train_packed, size_limit)
    
    if val_packed is not None and val_dir:
        print(f"(Dir {directory}) Writing validation set to {val_dir}", flush=True)
        _write_packed_dataset(val_dir, val_packed, size_limit)


def _load_index(local_dir: str) -> dict[str, Any]:
    index_path = pathlib.Path(local_dir) / "index.json"
    with open(index_path, "r") as f:
        index = json.load(f)

    if index["version"] != 2:
        raise ValueError(f"Invalid index version: {index['version']}, expected 2.")

    return index


def _from_numpy(record: dict[str, Any]) -> TokRecord:
    return {
        "tokens": record["tokens"].tolist(),
        "metadata": record["metadata"],
    }


def _to_numpy(record: TokRecord | PackTokRecord) -> dict[str, Any]:
    result = {
        "tokens": np.array(record["tokens"], dtype=np.int32),
        "metadata": record["metadata"],
    }
    # Add cu_seqlens if present (for packed records)
    if "cu_seqlens" in record:
        result["cu_seqlens"] = record["cu_seqlens"]
    return result


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


def _pack(
    records: Iterable[TokRecord],
    block_size: int,
    random_size: bool = False,
    desc: Optional[str] = None,
) -> Iterable[PackTokRecord]:
    def _empty() -> PackTokRecord:
        return {"tokens": [], "metadata": [], "cu_seqlens": [0]}

    def _append(
        curr: PackTokRecord, doc: TokRecord, size: int
    ) -> tuple[PackTokRecord, TokRecord]:
        leftover_size = size - len(curr["tokens"])
        doc_size = len(doc["tokens"])

        add_size = min(leftover_size, doc_size)
        
        # Only add metadata and cu_seqlens if we're adding a new document (not continuing one)
        if add_size > 0 and doc_size == len(doc["tokens"]):
            curr["metadata"].append(doc["metadata"])
        
        tokens_to_add = doc["tokens"][:add_size]
        curr["tokens"].extend(tokens_to_add)
        
        # Update cu_seqlens when we finish adding a complete document
        if add_size == doc_size:
            curr["cu_seqlens"].append(len(curr["tokens"]))

        doc["tokens"] = doc["tokens"][add_size:]

        return curr, doc

    curr = _empty()
    current_random_size, remaining_size = 0, 0
    min_doc_size = 12

    for doc in records:
        while len(doc["tokens"]) > 0:
            if random_size:
                if current_random_size == 0 and remaining_size == 0:
                    current_random_size = np.random.randint(
                        min_doc_size, block_size - (min_doc_size - 1)
                    )
                    remaining_size = block_size - current_random_size

                if current_random_size > 0:
                    curr, doc = _append(curr, doc, size=current_random_size)
                    if len(curr["tokens"]) >= current_random_size - min_doc_size:
                        yield curr
                        current_random_size = 0
                        curr = _empty()
                else:
                    curr, doc = _append(curr, doc, size=remaining_size)
                    if len(curr["tokens"]) >= remaining_size - min_doc_size:
                        yield curr
                        remaining_size = 0
                        curr = _empty()
            else:
                curr, doc = _append(curr, doc, size=block_size)
                if len(curr["tokens"]) == block_size:
                    yield curr
                    curr = _empty()

    if desc is not None:
        print(f"({desc}): leftovers:", len(curr["tokens"]))


def _build_histogram_streaming(
    input_dir: str, 
    block_size: int,
    head: Optional[int] = None
) -> Tuple[List[int], Dict[int, List[Tuple[int, int]]]]:
    """
    First pass: Build histogram and sequence locations without loading full data.
    
    Args:
        input_dir: Directory containing the tokenized dataset
        block_size: Maximum sequence length
        head: Optional limit on number of records to process
        
    Returns:
        histogram: List where histogram[i] = count of sequences with length i
        sequence_locations: Dict mapping seq_len -> [(shard_idx, record_idx), ...]
    """
    print(f"  Pass 1: Building histogram from {input_dir}...", flush=True)
    
    index = _load_index(input_dir)
    shards = index["shards"]
    
    histogram = [0] * (block_size + 1)
    sequence_locations = collections.defaultdict(list)
    
    total_records = 0
    
    for shard_idx, shard in enumerate(shards):
        reader = streaming.base.format.mds.MDSReader.from_json(input_dir, None, shard)
        
        for record_idx in range(reader.size):
            if head is not None and total_records >= head:
                break
                
            # Get only the length, not full tokens
            record = reader.get_item(record_idx)
            seq_len = min(len(record["tokens"]), block_size)
            
            histogram[seq_len] += 1
            sequence_locations[seq_len].append((shard_idx, record_idx))
            total_records += 1
        
        if head is not None and total_records >= head:
            break
    
    print(f"  Found {total_records} sequences", flush=True)
    
    return histogram, sequence_locations


def _fill_packing_strategy_streaming(
    assignments: List[List[int]],
    sequence_locations: Dict[int, List[Tuple[int, int]]],
    input_dir: str,
    block_size: int,
) -> List[Dict[str, Any]]:
    """
    Second pass: Fill bins with actual sequences using random access.
    
    Args:
        assignments: List of bins, each containing sequence lengths to pack together
        sequence_locations: Mapping from seq_len to [(shard_idx, record_idx), ...]
        input_dir: Directory containing the tokenized dataset
        block_size: Maximum sequence length
        
    Returns:
        List of packed sequences with tokens, metadata, and cu_seqlens
    """
    print(f"  Pass 2: Filling {len(assignments)} bins with sequences...", flush=True)
    
    # Load index and prepare readers
    index = _load_index(input_dir)
    shards = index["shards"]
    
    # Create readers for each shard (lazy loaded)
    readers = {}
    
    def get_reader(shard_idx: int):
        if shard_idx not in readers:
            readers[shard_idx] = streaming.base.format.mds.MDSReader.from_json(
                input_dir, None, shards[shard_idx]
            )
        return readers[shard_idx]
    
    # Randomize access order for each sequence length
    for seq_len, locations in sequence_locations.items():
        perm = np.random.permutation(len(locations))
        sequence_locations[seq_len] = [locations[i] for i in perm]
    
    # Fill bins
    output_data = []
    
    for bin_idx, assignment in enumerate(assignments):
        _tokens = []
        _metadata = []
        _cu_seqlens = [0]
        
        for seq_length in assignment:
            # Pop location and fetch the sequence
            shard_idx, record_idx = sequence_locations[seq_length].pop()
            reader = get_reader(shard_idx)
            record = reader.get_item(record_idx)
            
            # Extract tokens (truncate if needed)
            tokens = record["tokens"]
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            if len(tokens) > block_size:
                tokens = tokens[:block_size]
            
            _tokens.extend(tokens)
            _metadata.append(record.get("metadata", {}))
            _cu_seqlens.append(len(_tokens))
        
        output_data.append({
            "tokens": np.array(_tokens, dtype=np.int32),
            "metadata": _metadata,
            "cu_seqlens": _cu_seqlens,
        })
        
        if (bin_idx + 1) % 1000 == 0:
            print(f"    Packed {bin_idx + 1}/{len(assignments)} bins...", flush=True)
    
    # Verify all sequences were used
    unused = sum(len(locs) for locs in sequence_locations.values())
    if unused > 0:
        print(f"  Warning: {unused} sequences were not packed", flush=True)
    
    return output_data


def _pack_split_streaming(
    input_dir: str,
    block_size: int,
    packing_algorithm: str,
    head: Optional[int] = None,
    directory: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Pack a dataset split using two-pass streaming approach."""
    
    # Pass 1: Build histogram
    histogram, sequence_locations = _build_histogram_streaming(
        input_dir, block_size, head
    )
    
    # Determine packing strategy
    assignments, packing_metadata = create_packing_strategy(
        histogram,
        block_size,
        packing_algorithm,
    )
    
    # Display packing statistics
    dir_prefix = f"(Dir {directory}) " if directory else ""
    print(f"{dir_prefix}Packing Statistics:", flush=True)
    print(f"  Dataset max seq length:  {packing_metadata['dataset_max_seqlen']}", flush=True)
    print(f"  Max samples per bin:     {packing_metadata['max_samples_per_bin']}", flush=True)
    print(f"  Packing factor:          {packing_metadata['packing_factor']:.2f}x", flush=True)
    print(f"  Packing efficiency:      {packing_metadata['packing_efficiency']:.2f}%", flush=True)
    print(f"  Min packed seq length:   {packing_metadata['min_packed_seqlen']}", flush=True)
    
    # Pass 2: Fill bins
    packed_data = _fill_packing_strategy_streaming(
        assignments,
        sequence_locations,
        input_dir,
        block_size,
    )
    
    return packed_data


def _write_packed_dataset(
    output_dir: str,
    packed_data: List[Dict[str, Any]],
    size_limit: Optional[int | str],
):
    """Write packed dataset to MDS format."""
    columns = {
        "tokens": "ndarray:int32",
        "metadata": "json",
        "cu_seqlens": "json",
    }
    
    start = time.time()
    
    with streaming.MDSWriter(
        out=output_dir,
        columns=columns,
        size_limit=size_limit,
    ) as out:
        for i, record in enumerate(packed_data, start=1):
            out.write(record)
            if i % 1000 == 0:
                elapsed = time.time() - start
                records_per_second = i / max(int(elapsed), 1)
                print(
                    f"    Processed {i}/{len(packed_data)} records "
                    f"({records_per_second:.2f} Records/s)",
                    flush=True,
                )
                gc.collect()
    
    elapsed = time.time() - start
    fmt_elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"  Finished writing {len(packed_data)} records in {fmt_elapsed}", flush=True)


def _write_iterable(
    output_dir: str,
    records: Iterable[dict[str, Any]],
    size_limit: Optional[int | str],
    desc: Optional[str] = None,
):
    columns = {
        "tokens": "ndarray:int32",
        "metadata": "json",
        "cu_seqlens": "json",
    }
    start = time.time()
    prefix = f"({desc}): " if desc else ""

    def print_status(i: int, start: float):
        ellapsed = time.time() - start
        total_seconds = int(ellapsed)
        records_per_second = "Inf" if total_seconds == 0 else i / total_seconds
        msg = (
            f"{prefix}Processed {i} records in "
            f"{ellapsed // 3600}H:{(ellapsed % 3600) // 60}m:{ellapsed % 60}s ({records_per_second} Records/s)."
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
    print(f"{prefix}Finished writing {i} records.", flush=True)


if __name__ == "__main__":
    fire.Fire(pack_dataset)
