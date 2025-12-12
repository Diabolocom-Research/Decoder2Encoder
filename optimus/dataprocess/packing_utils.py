# Adapted from NVIDIA Megatron Bridge
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
from typing import Dict, List, Tuple

import numpy as np

PACKING_ALGOS = ["first_fit_decreasing", "first_fit_shuffle"]

logger = logging.getLogger(__name__)


def find_first_bin_that_fits(bins: List[List[int]], s: int, bin_size: int) -> int:
    """
    Finds the first bin in a list of bins that has enough space to fit a sequence of size 's'.

    Args:
      bins: A list of lists, where each inner list represents a bin and contains the current elements in that bin.
      s: The size of the sequence to be placed in a bin.
      bin_size: The maximum capacity of each bin.

    Returns:
      The index of the first bin that can fit the sequence 's', or -1 if no such bin exists.
    """
    for i, abin in enumerate(bins):
        if sum(abin) + s <= bin_size:
            return i
    return -1


def first_fit(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit algorithm.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, where each inner list represents a bin and contains the indices
        of the sequences assigned to that bin.
    """
    res = []
    for s in seqlens:
        first_bin = find_first_bin_that_fits(res, s, pack_size)
        if first_bin == -1:  # open a new bin
            res.append([s])
        else:
            res[first_bin].append(s)
    return res


def first_fit_decreasing(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit Decreasing algorithm.

    This is a variation of the First-Fit algorithm where the sequences are sorted by decreasing length before packing.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, similar to the output of the 'first_fit' function.
    """
    sorted_seqlens = sorted(seqlens, reverse=True)
    return first_fit(sorted_seqlens, pack_size)


def first_fit_shuffle(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit with Shuffling algorithm.

    This variation shuffles the order of the sequences before applying the First-Fit algorithm.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, similar to the output of the 'first_fit' function.
    """
    shuffled_seqlens = seqlens[:]
    np.random.shuffle(shuffled_seqlens)
    return first_fit(shuffled_seqlens, pack_size)


def create_packing_strategy(
    histogram: List[int], pack_size: int, packing_algorithm: str = "first_fit_shuffle"
) -> Tuple[List[List[int]], Dict[str, int]]:
    """
    Packs sequences into bins using the specified packing algorithm.

    This function takes the histogram of sequence lengths, desired pack size, and a string representing the packing
    algorithm to use. It then calls the corresponding function (e.g., 'first_fit_decreasing') and performs the
    packing process using only sequence lengths as input (without the actual sequences).

    Args:
          histogram: A list representing the histogram data (number of sequences for each length).
          pack_size: The maximum capacity of each bin.
          packing_algorithm: One of the supported packing algorithms from ['first_fit_decreasing', 'first_fit_shuffle']

    Returns:
          assignments: A list of lists, where each inner list represents a bin and contains the indices of the
                        sequence lengths assigned to that bin.
          pack_metadata: A dict that records packing metadata, for instance the max number of samples per bin.
    """
    all_seq_lens = []
    for i, count in enumerate(histogram):
        all_seq_lens.extend([i] * count)
    packing_fn = globals()[packing_algorithm]
    assignments: List[List[int]] = packing_fn(all_seq_lens, pack_size)
    packed_seq_lens = [sum(x) for x in assignments]
    packing_factor = len(all_seq_lens) / len(packed_seq_lens)

    max_seqlen = max(all_seq_lens)
    max_samples_per_bin = max([len(b) for b in assignments])
    min_packed_seqlen = min(packed_seq_lens)
    packing_efficiency = sum(packed_seq_lens) / len(packed_seq_lens) / pack_size * 100

    packing_metadata = {
        "dataset_max_seqlen": max_seqlen,
        "max_samples_per_bin": max_samples_per_bin,
        "packing_factor": round(packing_factor, 2),
        "packing_efficiency": round(packing_efficiency, 2),
        "pack_size": pack_size,
        "min_packed_seqlen": min_packed_seqlen,
    }

    return assignments, packing_metadata
