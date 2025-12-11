# experiments/run_sanity_generation.py

import torch
from transformers import AutoTokenizer

from moe_llm.config import SmolMoEConfig
from moe_llm.models import SmolMoELM
from moe_llm.checkpoints import download_smolmoe_weights
from moe_llm.generation import compare_generations


TEST_PROMPT = "Where is the Great Wall?"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load tokenizer – choose any compatible tokenizer
    #    (You can change this to whatever was used originally, e.g. a LLaMA-like tokenizer)
    tokenizer_name = "gpt2"  # placeholder – set to the actual tokenizer you used
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Build model and load weights
    config = SmolMoEConfig()
    model = SmolMoELM(config).to(device)

    # Use HF Hub helper (or local file if already downloaded)
    weights_path = download_smolmoe_weights(
        repo_id="dsouzadaniel/C4AI_SmolMoELM",
        filename="trial_weights.pt",
        local_dir=".",
    )

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)

    # 3. Run a sanity generation
    compare_generations(
        prompt=TEST_PROMPT,
        tokenizer=tokenizer,
        model_a=model,
        model_b=None,
        max_new_tokens=50,
        device=device,
    )


if __name__ == "__main__":
    main()
