# experiments/run_continued_pretraining_entropy_selection.py

import math
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

from moe_llm.config import from_hf_causal_lm_config
from moe_llm.models import SmolMoELM
from moe_llm.data import build_dataset, build_dataloaders
from moe_llm.data_selection import stratified_fixed_total_token_entropy
from moe_llm.training import continued_pretraining
from moe_llm.generation import compare_generations
from moe_llm.metrics import plot_metrics


TEST_PROMPT = "Where is the Great Wall?"


def main(
    checkpoint: str = "HuggingFaceTB/SmolLM-135M",  
    steps: int = 100,
    report_every: int = 10,
    batch_size: int = 4,
    bf16: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dense model + tokenizer
    dense_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Build dataset and apply entropy-based selection on the raw docs
    raw_ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
    keep_indices, report = stratified_fixed_total_token_entropy(
        raw_ds,
        total_samples=1000,
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        group_cols=("audience", "format", "seed_data"),
        w_text=1.0,
        w_prompt=0.25,
        min_text_chars=50,
        min_prompt_chars=10,
        even_strategy="even_then_fill",
    )
    print("Selected:", len(keep_indices))
    print(
        "Min/Max per-group:",
        report["quota_summary"]["min_quota"],
        report["quota_summary"]["max_quota"],
    )

    sub_ds = raw_ds.select(keep_indices)

    # Now reuse the same blockification/tokenization logic on the filtered subset
    # (we treat it as if it were our "raw dataset" limited to these 1000 chosen examples)

    from moe_llm.data import build_dataset as _build_dataset

    train_ds, val_ds = _build_dataset(
        dataset_id="HuggingFaceTB/cosmopedia-100k",
        subset=None,
        split="train",
        tokenizer=tokenizer,
        block_size=256,
        val_fraction=0.2,
        max_samples=len(sub_ds),
        seed=789,
    )

    train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size, device)

    # 3. MoE model (dense → MoE upcycling)
    from moe_llm.upcycling import upcycle_from_dense

    hf_cfg = dense_model.config
    moe_cfg = from_hf_causal_lm_config(hf_cfg, num_experts=3, num_experts_per_token=1)
    moe_model = SmolMoELM(moe_cfg).to(device)
    upcycle_from_dense(dense_model, moe_model, layers="all")

    # 4. Optimizer + scheduler (same as before)
    peak_lr = 1e-2
    weight_decay = 0.01

    def invsqrt_lambda(step: int) -> float:
        return 1.0 / math.sqrt(max(step, 1))

    opt = Adafactor(
        moe_model.parameters(),
        lr=peak_lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        weight_decay=weight_decay,
        clip_threshold=1.0,
        eps=(1e-30, 1e-3),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=invsqrt_lambda)

    # 5. Continued pretraining with *entropy-selected* subset
    training_metrics, moe_metrics = continued_pretraining(
        moe_model,
        train_loader,
        val_loader,
        opt,
        scheduler,
        device=device,
        steps=steps,
        report_every=report_every,
        bf16=bf16,
        accum_steps=16,
        lb_coef=0.01,
    )

    # 6. Plot metrics
    if training_metrics["Train Loss"]:
        x_vals = [report_every * i for i in range(1, len(training_metrics["Train Loss"]) + 1)]
        plot_metrics(training_metrics, x_vals=x_vals, suptitle="Training Metrics", save_dir="runs/exp2_with_selection/plots", filename="training_metrics.png", show=False)
        plot_metrics(moe_metrics, x_vals=x_vals, suptitle="MoE Metrics", save_dir="runs/exp2_with_selection/plots", filename="moe_metrics.png", show=False)

    # 7. Final generation sanity check
    moe_model.to("cpu").eval()
    dense_model.to("cpu").eval()

    compare_generations(
        prompt=TEST_PROMPT,
        tokenizer=tokenizer,
        model_a=dense_model,
        model_b=moe_model,
        max_new_tokens=50,
        device="cpu",
    )


if __name__ == "__main__":
    main()
