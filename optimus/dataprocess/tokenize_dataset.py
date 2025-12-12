"""Tokenize datasets using HuggingFace tokenizers with multiprocessing support."""

import gc
import importlib
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Protocol, TypedDict

import fire
import numpy as np
from dateutil import parser
from streaming import MDSWriter
from streaming.base.util import merge_index
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Type definitions
Metadata = dict[str, Any] | list[Any]


class Record(TypedDict):
    text: str
    metadata: Metadata


class TokenizedRecord(TypedDict):
    tokens: list[int]
    metadata: Metadata


class GetRecordsFunc(Protocol):
    def __call__(self, file: str) -> Iterable[list[Record]]: ...


class TokenizationError(Exception):
    pass


def parse_time_string(time_str: Optional[str]) -> Optional[int]:
    if time_str is None:
        return None

    try:
        time_obj = parser.parse(time_str)
    except parser.ParserError as e:
        raise ValueError(f"Invalid time format. Expected HH:MM:SS, got: {time_str}") from e

    return int(time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second)


def monitor_futures_with_timeout(
    executor: ProcessPoolExecutor,
    futures: list,
    timeout: Optional[str],
) -> None:
    if not timeout:
        return

    timeout_seconds = parse_time_string(timeout)
    start_time = time.time()

    while not all(f.done() for f in futures):
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"Timeout reached after {elapsed:.1f}s, canceling remaining tasks.")
            executor.shutdown(wait=False, cancel_futures=True)
            break
        time.sleep(10)


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerBase:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    except Exception as e:
        raise TokenizationError(f"Failed to load tokenizer from '{tokenizer_path}': {e}") from e


def encode_texts(
    texts: list[str | list[dict]],
    tokenizer: PreTrainedTokenizerBase,
) -> list[list[int]]:
    try:
        if not texts:
            return []

        if isinstance(texts[0], str):
            result = tokenizer(texts, add_special_tokens=False)
            return result["input_ids"]
        else:
            result = tokenizer.apply_chat_template(
                texts,
                tokenize=True,
                add_generation_prompt=False,
            )
            return result

    except Exception as e:
        raise TokenizationError(f"Failed to encode texts: {e}") from e


def process_worker(
    input_files: list[str],
    tokenizer_path: str,
    output_dir: str,
    size_limit: int | str,
    get_records_fn: GetRecordsFunc,
    head: Optional[int] = None,
) -> int:
    """
    Worker process to tokenize a subset of input files.

    This function uses lazy evaluation and streaming to minimize memory usage:
    - Uses map() instead of list comprehensions to avoid materializing large lists
    - Uses itertools.chain to lazily flatten nested iterators
    - Processes records one at a time through the pipeline
    - Calls gc.collect() periodically to free memory

    Args:
        input_files: List of input file paths to process
        tokenizer_path: Path to tokenizer or model name
        output_dir: Output directory for this worker
        size_limit: Maximum size per output shard
        get_records_fn: Function to extract records from files
        head: Maximum number of batches to process (for testing)

    Returns:
        Total number of tokens processed
    """
    worker_id = Path(output_dir).name
    print(f"[{worker_id}] Worker started", flush=True)

    tokenizer = load_tokenizer(tokenizer_path)

    data_per_file = map(get_records_fn, input_files)
    batches = itertools.chain.from_iterable(data_per_file)

    if head is not None:
        batches = itertools.islice(batches, head)

    def _tokenize(batch: list[Record]) -> Iterator[TokenizedRecord]:
        texts = [r["text"] for r in batch]
        metadata_list = [r.get("metadata", {}) for r in batch]
        tokens_list = encode_texts(texts, tokenizer)
        return ({"tokens": t, "metadata": m} for t, m in zip(tokens_list, metadata_list))

    tokenized_batches = map(_tokenize, batches)
    tokenized = itertools.chain.from_iterable(tokenized_batches)

    def to_numpy(record: TokenizedRecord) -> dict[str, Any]:
        record["tokens"] = np.array(record["tokens"], dtype=np.int32)
        return record

    tokenized = map(to_numpy, tokenized)

    columns = {
        "tokens": "ndarray:int32",
        "metadata": "json",
    }

    start_time = time.time()
    total_tokens = 0

    with MDSWriter(out=output_dir, columns=columns, size_limit=size_limit) as writer:
        for i, record in enumerate(tokenized, start=1):
            if record["tokens"].size:
                total_tokens += record["tokens"].size
                writer.write(record)

            if i % 10000 == 0:
                elapsed = time.time() - start_time
                records_per_sec = i / max(int(elapsed), 1)
                print(
                    f"[{worker_id}] Processed {i} records, "
                    f"{records_per_sec:.2f} Records/s, {total_tokens} tokens.",
                    flush=True,
                )
                gc.collect()

    return total_tokens


def tokenize_dataset(
    input_dir: str,
    tokenizer: str,
    dataset: str,
    output_dir: str = "output",
    size_limit: Optional[int | str] = "64MB",
    num_workers: int | str = 1,
    head: Optional[int] = None,
    timeout: Optional[str] = None,
    read_files_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """
    Tokenize a dataset using HuggingFace tokenizers with parallel processing.

    Args:
        input_dir: Path to directory containing input data files
        tokenizer: HuggingFace tokenizer name or path
        dataset: Dataset loader module name (must exist in optimus.dataprocess.dataset/)
        output_dir: Output directory for tokenized data (must not exist)
        size_limit: Maximum size per output shard (e.g., "64MB", "1GB")
        num_workers: Number of parallel workers (or "max" for CPU count - 1)
        head: Limit number of batches per worker (for testing)
        timeout: Maximum runtime in HH:MM:SS format
        read_files_kwargs: Additional arguments passed to dataset loader
    """
    start_time = time.time()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(file_dir, "dataset")
    assert dataset + ".py" in os.listdir(dataset_dir), f"{dataset}.py module not found."
    dataset_module = importlib.import_module(f"optimus.dataprocess.dataset.{dataset}")

    read_files_kwargs = read_files_kwargs or {}

    assert (
        num_workers == "max" or num_workers > 0
    ), "num_workers must be greater than 0."
    num_workers = os.cpu_count() - 1 if num_workers == "max" else num_workers

    assert not os.path.exists(
        output_dir
    ), f"Output directory '{output_dir}' already exists."

    print(
        f"Input directory: {input_dir}\nTokenizer: {tokenizer}\nDataset: {dataset}\n"
        f"Size limit: {size_limit}\nNum workers: {num_workers}\nTimeout: {timeout}"
    )

    inputs = dataset_module.get_files(input_dir, **read_files_kwargs)
    assert inputs, "No data files found in the input directory."

    if num_workers > len(inputs):
        print(f"Reduced num_workers to {len(inputs)} to match the number of inputs.")
        num_workers = len(inputs)

    output_dirs = (
        [os.path.join(output_dir, str(i)) for i in range(num_workers)]
        if num_workers > 1
        else [output_dir]
    )
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)

    input_subsets = np.array_split(inputs, num_workers)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_worker,
                subset,
                tokenizer,
                output_dirs[i],
                size_limit,
                dataset_module.get_text,
                head,
            )
            for i, subset in enumerate(input_subsets)
        ]

        if timeout:
            monitor_futures_with_timeout(executor, futures, timeout)

    total_tokens = sum(future.result() for future in futures if future.done())

    if num_workers > 1:
        merge_index(output_dir, keep_local=True)

    elapsed = time.time() - start_time
    metadata = {
        "tokenizer": tokenizer,
        "input_dir": input_dir,
        "dataset": dataset,
        "size_limit": size_limit,
        "num_workers": num_workers,
        "total_tokens": total_tokens,
        "Runtime": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Tokenization completed.")


if __name__ == "__main__":
    fire.Fire(tokenize_dataset)
