# src/moe_llm/moe_metrics.py
import math
import torch

from .utils import labelthis
from .models import SmolMoELM  # type hint only


def normalized_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalized entropy H(p) / log(E) for distributions over E experts.
    """
    p = p.clamp_min(eps)
    H = -(p * p.log()).sum(dim=-1)
    E = p.size(-1)
    return H / math.log(E)


@torch.no_grad()
def expert_balance_score(model: SmolMoELM) -> float:
    """
    MoE-specific diagnostic:
      Compute normalized entropy of expert utilization across layers,
      then average across layers.
    """
    util_per_layer, _ = model.get_expert_utilization()  # [L, E]
    util_per_layer = torch.nan_to_num(util_per_layer, nan=0.0)

    p = util_per_layer / (util_per_layer.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    B_layer = normalized_entropy(p).mean().item()
    return B_layer


@labelthis("Balanced PPL")
@torch.no_grad()
def custom_moe_metric(model: SmolMoELM, ce_mean: float, beta: float = 1.0) -> float:
    """
    Balanced PPL = exp(CE) * exp(beta * (1 - B))

    Where:
      - CE is cross-entropy (validation loss)
      - B is the expert balance score (normalized entropy of utilization)
        B ~ 1: well-balanced usage
        B << 1: highly imbalanced usage

    Lower is better: we want low perplexity with well-balanced experts.
    """
    ppl = math.exp(ce_mean)
    B = expert_balance_score(model)
    metric = float(ppl * math.exp(beta * (1.0 - B)))
    return metric
