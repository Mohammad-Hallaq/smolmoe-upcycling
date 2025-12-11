# src/moe_llm/upcycling.py

from typing import Iterable, Union

import torch
from torch import nn

from .models import SmolMoELM


@torch.no_grad()
def copy_shared_weights_from_dense(moe_model: SmolMoELM, dense_model: nn.Module) -> None:
    """
    Copy shared components (embeddings, norms, LM head, attention weights, layer norms)
    from a dense causal LM into the MoE model.

    Assumes a LLaMA-like HF model with keys:
      - model.embed_tokens.weight
      - model.norm.weight
      - lm_head.weight
      - model.layers.{i}.self_attn.(q_proj,k_proj,v_proj,o_proj).weight
      - model.layers.{i}.input_layernorm.weight
      - model.layers.{i}.post_attention_layernorm.weight
    """
    sd = dense_model.state_dict()

    # Embeddings / final norm / head
    moe_model.model.embed_tokens.weight.copy_(sd["model.embed_tokens.weight"])
    moe_model.model.norm.weight.copy_(sd["model.norm.weight"])
    moe_model.lm_head.weight.copy_(sd["lm_head.weight"])

    # Per-layer attention + norms
    for i, dec in enumerate(moe_model.model.layers):
        dec.self_attn.W_query.weight.copy_(
            sd[f"model.layers.{i}.self_attn.q_proj.weight"]
        )
        dec.self_attn.W_key.weight.copy_(
            sd[f"model.layers.{i}.self_attn.k_proj.weight"]
        )
        dec.self_attn.W_value.weight.copy_(
            sd[f"model.layers.{i}.self_attn.v_proj.weight"]
        )
        dec.self_attn.W_output.weight.copy_(
            sd[f"model.layers.{i}.self_attn.o_proj.weight"]
        )
        dec.pre_attn_rmsnorm.weight.copy_(
            sd[f"model.layers.{i}.input_layernorm.weight"]
        )
        dec.pre_moe_rmsnorm.weight.copy_(
            sd[f"model.layers.{i}.post_attention_layernorm.weight"]
        )


@torch.no_grad()
def upcycle_dense_ffn_into_moe_banks(
    moe_layer: nn.Module,
    up_W: torch.Tensor,
    gate_W: torch.Tensor,
    down_W: torch.Tensor,
    zero_router: bool = True,
) -> None:
    """
    Populate a single MoE layer's banks using dense FFN weights.

    moe_layer: an MoE instance with attributes up_bank, gate_bank, down_bank, gate.
      - up_bank:  [E, D, H]
      - gate_bank:[E, D, H]
      - down_bank:[E, H, D]

    up_W, gate_W, down_W:
      dense FFN weights, e.g. from a LLaMA-like MLP:
        - up_proj.weight   (H, D) or (D, H)
        - gate_proj.weight (H, D) or (D, H)
        - down_proj.weight (D, H) or (H, D)
    """
    E, D, H = moe_layer.up_bank.shape

    # Normalize shapes to [D, H] for up/gate and [H, D] for down
    if up_W.shape == (H, D):
        up_W = up_W.T
    if gate_W.shape == (H, D):
        gate_W = gate_W.T
    if down_W.shape == (D, H):
        down_W = down_W.T

    # Replicate dense FFN into all experts (initially identical experts)
    for e in range(E):
        moe_layer.up_bank[e].copy_(up_W)
        moe_layer.gate_bank[e].copy_(gate_W)
        moe_layer.down_bank[e].copy_(down_W)

    # Optional: initialize router to zeros so routing is initially uniform
    if zero_router:
        moe_layer.gate.weight.zero_()


@torch.no_grad()
def upcycle_from_dense(
    dense_model: nn.Module,
    moe_model: SmolMoELM,
    layers: Union[str, Iterable[int]] = "all",
) -> None:
    """
    Upcycle a dense model into the MoE model:

      1. Copy shared weights (embeddings, norms, attention).
      2. For each target layer, map dense FFN weights into MoE expert banks.

    Parameters
    ----------
    dense_model : HF causal LM (LLaMA-like)
    moe_model   : SmolMoELM with matching depth/width
    layers      : "all" or iterable of layer indices
    """
    sd = dense_model.state_dict()
    copy_shared_weights_from_dense(moe_model, dense_model)

    n_layers = len(moe_model.model.layers)
    target_layers = range(n_layers) if layers == "all" else layers

    for i in target_layers:
        dec = moe_model.model.layers[i]
        up = sd[f"model.layers.{i}.mlp.up_proj.weight"]
        gate = sd[f"model.layers.{i}.mlp.gate_proj.weight"]
        down = sd[f"model.layers.{i}.mlp.down_proj.weight"]
        upcycle_dense_ffn_into_moe_banks(dec.moe, up, gate, down)
