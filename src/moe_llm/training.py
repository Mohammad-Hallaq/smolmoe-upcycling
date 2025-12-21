# src/moe_llm/training.py
import time
from typing import Dict, List, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .losses import causal_lm_loss
from .moe_metrics import custom_moe_metric
from .utils import pretty_dt


@torch.no_grad()
def evaluate_loss(
    model,
    val_loader: DataLoader,
    device: torch.device,
    bf16: bool = True,
) -> float:
    model.eval()
    losses: List[float] = []

    amp_device_type = "cuda" if device.type == "cuda" else "cpu"

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=amp_device_type,
            dtype=torch.bfloat16,
            enabled=bf16 and device.type == "cuda",
        ):
            logits = model(input_ids=input_ids, attention_mask=attn_mask)["logits"]
            l = causal_lm_loss(logits, input_ids, attn_mask)
        losses.append(l.item())

    model.train()
    return sum(losses) / max(len(losses), 1)


def continued_pretraining(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    opt: Optimizer,
    scheduler,
    device: torch.device,
    steps: int = 100,
    report_every: int = 10,
    bf16: bool = True,
    accum_steps: int = 16,
    lb_coef: float = 0.01,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Continued pretraining loop focusing on training the MoE router.

    Returns
    -------
    training_metrics : dict of lists
        'Train Loss', 'Eval Loss', 'Load Balancing Loss'
    moe_metrics : dict of lists
        {custom_moe_metric.label: [values]}
    """
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"

    # Initial sanity check
    val_loss = evaluate_loss(model, val_loader, device, bf16=bf16)
    moe_value = custom_moe_metric(model, val_loss)
    print(f"[Before Training : Sanity Check] {custom_moe_metric.label}: {moe_value:.1f}\n")

    training_metrics: Dict[str, List[float]] = {
        "Train Loss": [],
        "Eval Loss": [],
        "Load Balancing Loss": [],
    }
    moe_metrics: Dict[str, List[float]] = {custom_moe_metric.label: []}

    global_step = 0
    micro = 0
    t0 = time.time()

    window_lm_sum = 0.0
    window_lb_sum = 0.0
    running_train_loss = 0.0
    running_lb_loss = 0.0

    model.train()

    while global_step < steps:
        for batch in train_loader:
            if global_step >= steps:
                break

            micro += 1

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=amp_device_type,
                dtype=torch.bfloat16,
                enabled=bf16 and device.type == "cuda",
            ):
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                logits = out["logits"]
                lm = causal_lm_loss(logits, input_ids, attn_mask)

                _, lb = model.get_expert_utilization()

                # NaN / inf guards
                if torch.isnan(lm) or torch.isinf(lm):
                    print(f"WARNING: LM loss is {lm.item()}, skipping batch")
                    continue
                if torch.isnan(lb) or torch.isinf(lb):
                    print(f"WARNING: LB loss is {lb.item()}, setting to 0")
                    lb = torch.tensor(0.0, device=lb.device)

                loss = (lm + lb_coef * lb) / accum_steps

            loss.backward()

            window_lm_sum += lm.detach().item()
            window_lb_sum += lb.detach().item()

            if micro % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

                global_step += 1

                running_train_loss += window_lm_sum / accum_steps
                running_lb_loss += window_lb_sum / accum_steps

                window_lm_sum = 0.0
                window_lb_sum = 0.0

                # Reporting
                if global_step % report_every == 0:
                    time_taken = time.time() - t0

                    train_ce_mean = running_train_loss / report_every
                    lb_mean = running_lb_loss / report_every

                    val_loss = evaluate_loss(model, val_loader, device, bf16=bf16)

                    training_metrics["Train Loss"].append(train_ce_mean)
                    training_metrics["Eval Loss"].append(val_loss)
                    training_metrics["Load Balancing Loss"].append(lb_mean)

                    ce_mean = sum(training_metrics["Eval Loss"]) / len(
                        training_metrics["Eval Loss"]
                    )
                    moe_value = custom_moe_metric(model, ce_mean)
                    moe_metrics[custom_moe_metric.label].append(moe_value)

                    print(
                        f"Step {global_step}/{steps} | "
                        f"Train Loss: {train_ce_mean:.3f} | "
                        f"Eval Loss: {val_loss:.3f} | "
                        f"LB Loss: {lb_mean:.3f} | "
                        f"Time Taken: {pretty_dt(time_taken)}"
                    )
                    print(
                        f"Step {global_step}/{steps} | "
                        f"{custom_moe_metric.label}: {moe_value:.1f}\n"
                    )
                    print("***" * 30)

                    running_train_loss = 0.0
                    running_lb_loss = 0.0
                    t0 = time.time()

    return training_metrics, moe_metrics
