# experiments/run_continued_pretraining.py

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

from moe_llm.config import from_hf_causal_lm_config
from moe_llm.models import SmolMoELM
from moe_llm.data import build_dataset, build_dataloaders
from moe_llm.training import continued_pretraining
from moe_llm.moe_metrics import custom_moe_metric
from moe_llm.generation import compare_generations
from moe_llm.upcycling import upcycle_from_dense
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

    # 1. Load dense model and tokenizer
    dense_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Build MoE from dense config and upcycle
    hf_cfg = dense_model.config
    moe_cfg = from_hf_causal_lm_config(hf_cfg, num_experts=3, num_experts_per_token=1)
    moe_model = SmolMoELM(moe_cfg).to(device)

    upcycle_from_dense(dense_model, moe_model, layers="all")

    # 3. Build dataset + dataloaders
    train_ds, val_ds = build_dataset(
        dataset_id="HuggingFaceTB/cosmopedia-100k",
        subset=None,
        split="train",
        tokenizer=tokenizer,
        block_size=256,
        val_fraction=0.2,
        max_samples=1000,
        seed=789,
    )

    train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size, device)
    print(f"Train Dataset Batches : {len(train_loader)}")
    print(f"Validation Dataset Batches : {len(val_loader)}")

    # 4. Optimizer + scheduler (Adafactor + inverse sqrt)
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

    # 5. Continued pretraining
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
        plot_metrics(training_metrics, x_vals=x_vals, suptitle="Training Metrics", save_dir="runs/exp1/plots", filename="training_metrics.png", show=False)
        plot_metrics(moe_metrics, x_vals=x_vals, suptitle="MoE Metrics", save_dir="runs/exp1/plots", filename="moe_metrics.png", show=False)

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
