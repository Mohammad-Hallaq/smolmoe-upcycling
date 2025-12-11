# src/moe_llm/losses.py
import math
import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Standard next-token causal LM loss with masking.

    logits: [B, T, V]
    input_ids: [B, T]
    """
    B, T, V = logits.shape
    tgt = input_ids[:, 1:].contiguous()     # next-token targets
    pred = logits[:, :-1, :].contiguous()   # predictions up to T-1

    if attention_mask is not None:
        mask = attention_mask[:, 1:].contiguous().to(dtype=pred.dtype)
    else:
        mask = torch.ones_like(tgt, dtype=pred.dtype)

    tgt_masked = tgt.masked_fill(mask == 0, ignore_index)

    loss = F.cross_entropy(
        pred.view(-1, V),
        tgt_masked.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, -1)

    loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    return loss
