# experiments/run_upcycling_demo.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_llm.config import from_hf_causal_lm_config
from moe_llm.models import SmolMoELM
from moe_llm.generation import compare_generations
from moe_llm.upcycling import upcycle_from_dense


TEST_PROMPT = "Where is the Great Wall?"


def main(
    checkpoint: str = "HuggingFaceTB/SmolLM-135M", 
    max_new_tokens: int = 50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dense model + tokenizer
    dense_model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Build MoE model config from dense config
    hf_cfg = dense_model.config
    moe_cfg = from_hf_causal_lm_config(
        hf_cfg,
        num_experts=3,
        num_experts_per_token=1,
    )

    moe_model = SmolMoELM(moe_cfg).to(device)
    moe_model.eval()

    print("\n=== Before Upcycling (MoE randomly initialized) ===")
    compare_generations(
        prompt=TEST_PROMPT,
        tokenizer=tokenizer,
        model_a=dense_model,
        model_b=moe_model,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    # 3. Upcycle dense weights into MoE
    print("\n=== Running Upcycling from dense → MoE ===")
    upcycle_from_dense(dense_model, moe_model, layers="all")

    print("\n=== After Upcycling (MoE vs Dense) ===")
    compare_generations(
        prompt=TEST_PROMPT,
        tokenizer=tokenizer,
        model_a=dense_model,
        model_b=moe_model,
        max_new_tokens=max_new_tokens,
        device=device,
    )


if __name__ == "__main__":
    main()
