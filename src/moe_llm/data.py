# src/moe_llm/data.py
from typing import Tuple, Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def build_dataset(
    dataset_id: str,
    subset: Optional[str],
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    max_samples: int = 1000,
    text_column: str = "text",
    val_fraction: Optional[float] = None,
    seed: int = 42,
):
    """
    Build a tokenized, blockified LM dataset, optionally split into train/val.

    - Adds EOS to each sequence
    - Chops documents into fixed-length blocks (block_size)
    """
    if subset:
        ds = load_dataset(dataset_id, subset, split=split)
    else:
        ds = load_dataset(dataset_id, split=split)

    ds = ds.select(range(max_samples))

    eos_id = tokenizer.eos_token_id

    def tok(batch):
        out = tokenizer(
            batch[text_column],
            add_special_tokens=False,
            return_attention_mask=True,
        )
        out["input_ids"] = [ids + [eos_id] for ids in out["input_ids"]]
        out["attention_mask"] = [m + [1] for m in out["attention_mask"]]
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
        }

    ds = ds.map(
        tok,
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in ("input_ids", "attention_mask")],
    )

    def group_per_doc(batch):
        out_ids = []
        for ids in batch["input_ids"]:
            L = len(ids)
            n = (L // block_size) * block_size
            for i in range(0, n, block_size):
                out_ids.append(ids[i : i + block_size])
        return {
            "input_ids": out_ids,
            "attention_mask": [[1] * len(o) for o in out_ids],
        }

    ds = ds.map(group_per_doc, batched=True)

    if val_fraction and 0.0 < val_fraction < 1.0:
        ds = ds.train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
        train_ds, val_ds = ds["train"], ds["test"]
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return train_ds, val_ds

    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds


def build_dataloaders(
    train_ds,
    val_ds,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Simple DataLoader wrappers."""
    pin = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin,
    )
    return train_loader, val_loader
