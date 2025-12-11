# src/moe_llm/generation.py
from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from .utils import timed


@timed
def greedy_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 32,
    device: str | torch.device = "cpu",
) -> str:
    """
    Simple greedy token-by-token generation loop.

    Parameters
    ----------
    model : causal LM (HF-style or compatible)
    tokenizer : tokenizer with eos_token_id
    prompt : str
    max_new_tokens : int

    Returns
    -------
    generated_text : str
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            output = model(**inputs)
        next_id = torch.argmax(output["logits"][0, -1]).item()
        generated_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        # Append new token to input
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([[next_id]], device=device)],
            dim=1,
        )
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def compare_generations(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    model_a,
    model_b: Optional[PreTrainedModel] = None,
    max_new_tokens: int = 32,
    device: str | torch.device = "cpu",
) -> None:
    """Print side-by-side generations from two models (or just one if model_b is None)."""
    print()
    print(f"{'>' * 20}\n\tPrompt\n{'<' * 20}\n{prompt}\n")

    print(f"{'>' * 30}\n\tModel A Generation\n{'<' * 30}")
    gen_a = greedy_generate(model_a, tokenizer, prompt, max_new_tokens, device=device)
    print(gen_a)

    if model_b is not None:
        print("\n")
        print(f"{'>' * 30}\n\tModel B Generation\n{'<' * 30}")
        gen_b = greedy_generate(model_b, tokenizer, prompt, max_new_tokens, device=device)
        print(gen_b)
