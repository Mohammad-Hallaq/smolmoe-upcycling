# src/moe_llm/config.py
from dataclasses import dataclass


@dataclass
class SmolMoEConfig:
    vocab_size: int = 49_152
    hidden_size: int = 576
    intermediate_size: int = 1_536
    num_hidden_layers: int = 30
    num_heads: int = 9
    kv_heads: int = 3
    num_experts: int = 3
    num_experts_per_token: int = 1
    rope_theta: float = 10000.0
    moe_noise_std: float = 1e-1


def from_hf_causal_lm_config(
    hf_cfg,
    num_experts: int = 3,
    num_experts_per_token: int = 1,
) -> SmolMoEConfig:
    """
    Construct a SmolMoEConfig from a Hugging Face causal LM config.
    Assumes LLaMA-like naming: model.embed_tokens, model.layers, etc.
    """
    return SmolMoEConfig(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        intermediate_size=hf_cfg.intermediate_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_heads=hf_cfg.num_attention_heads,
        kv_heads=getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
    )
