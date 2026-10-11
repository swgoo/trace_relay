"""Packed token streams. Every next-token target is visited once per epoch."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokens(Dataset):
    def __init__(self, directory, split, sequence_length):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        directory = Path(directory)
        self.tokens = np.memmap(directory / f"{split}.bin", dtype="<u2", mode="r")
        self.words = np.memmap(directory / f"{split}.words.bin", dtype="u1", mode="r")
        if len(self.tokens) < 2 or len(self.tokens) != len(self.words):
            raise ValueError(f"Invalid or empty {split} token stream")
        self.sequence_length = sequence_length

    def __len__(self):
        return math.ceil((len(self.tokens) - 1) / self.sequence_length)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        count = min(self.sequence_length, len(self.tokens) - start - 1)
        inputs = torch.zeros(self.sequence_length, dtype=torch.long)
        targets = torch.full_like(inputs, -100)
        inputs[:count] = torch.from_numpy(self.tokens[start:start + count].astype(np.int64))
        targets[:count] = torch.from_numpy(self.tokens[start + 1:start + count + 1].astype(np.int64))
        mask = torch.arange(self.sequence_length) < count
        return {"input_ids": inputs, "labels": targets, "attention_mask": mask,
                "words": int(self.words[start + 1:start + count + 1].sum())}
