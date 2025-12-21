# src/moe_llm/models.py
import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import SmolMoEConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.device != q.device:
        cos = cos.to(q.device)
        sin = sin.to(q.device)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryEmbedder(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        self.freq = 1/(base ** (torch.arange(0, dim, 2, dtype=torch.int64).float()/dim))

    @torch.no_grad()
    def forward(self,x):
        pos = torch.arange(x.shape[-2],dtype=torch.long)
        angles = torch.einsum('f,p->pf', self.freq, pos.float()).unsqueeze(dim=0)

        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos(), emb.sin()


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) layer with gated experts:
      - A shared gating network produces expert logits per token.
      - A bank of expert parameters implements (gated) feedforward transformations.
      - Supports top-k expert selection per token.
    """

    def __init__(
        self,
        num_experts_per_token: int,
        num_experts: int,
        emb_dim: int,
        moe_dim: int,
        dtype: torch.dtype = torch.float32,
        noise_std: float = 1e-1,
    ):
        super().__init__()
        self.k = int(num_experts_per_token)
        self.E = int(num_experts)
        self.D = int(emb_dim)
        self.H = int(moe_dim)
        self.noise_std = noise_std

        self.gate = nn.Linear(self.D, self.E, bias=False, dtype=dtype)

        # Expert parameter banks: [E, D, H] and [E, H, D]
        self.gate_bank = nn.Parameter(torch.empty(self.E, self.D, self.H, dtype=dtype))
        self.up_bank = nn.Parameter(torch.empty(self.E, self.D, self.H, dtype=dtype))
        self.down_bank = nn.Parameter(torch.empty(self.E, self.H, self.D, dtype=dtype))

        # Buffers for diagnostics (expert utilization + aux loss)
        self._expert_utilization = None
        self._aux_lb = None

    def expert_utilization(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-expert utilization and a Switch-Transformer-style load-balancing loss.
        Ref: https://arxiv.org/abs/2101.03961
        """
        probs = logits.softmax(dim=-1)              # [B, T, E]
        P_i = probs.mean(dim=(0, 1))                # [E]

        selected = torch.argmax(logits, dim=-1)     # [B, T]
        selected = F.one_hot(selected, num_classes=self.E)  # [B, T, E]
        load = selected.float().mean(dim=(0, 1))    # [E]

        aux_lb = self.E * torch.sum(P_i * load)

        self._expert_utilization = load
        self._aux_lb = aux_lb

        return load, aux_lb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        Returns: [B, T, D]
        """
        B, T, D = x.shape
        assert D == self.D, f"Expected emb_dim={self.D}, got {D}"

        logits = self.gate(x)  # [B, T, E]

        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # Top-k routing
        probs = logits.softmax(-1)                                   # [B, T, E]
        selected_probs, selected_idx = torch.topk(
            probs, self.k, dim=-1
        )                                                           # [B, T, k]
        selected_probs = selected_probs / (
            selected_probs.sum(dim=-1, keepdim=True) + 1e-8
        )                                                           # normalize

        # Expert computations: [B, T, E, H]
        a = torch.einsum("btd,edh->bteh", x, self.gate_bank)
        u = torch.einsum("btd,edh->bteh", x, self.up_bank)

        h = F.silu(a) * u                                           # gated activation
        y_all = torch.einsum("bteh,ehd->bted", h, self.down_bank)   # [B, T, E, D]

        # Gather only selected experts
        gather_idx = selected_idx.view(B, T, self.k, 1).expand(-1, -1, -1, D)
        y_selected = torch.gather(y_all, dim=2, index=gather_idx)   # [B, T, k, D]
        y = (y_selected * selected_probs.unsqueeze(-1)).sum(dim=2)  # [B, T, D]

        self.expert_utilization(logits)
        return y


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class RopeAttention(nn.Module):
    def __init__(self, config: SmolMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.kv_heads = config.kv_heads

        self.W_query = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.W_key = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_value = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedder(
            base=config.rope_theta,
            dim=config.hidden_size // self.num_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, q, _ = hidden_states.size()

        q_states = self.W_query(hidden_states)
        k_states = self.W_key(hidden_states)
        v_states = self.W_value(hidden_states)

        q_states = q_states.view(b, q, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = k_states.view(b, q, self.kv_heads, self.head_dim).transpose(1, 2)
        v_states = v_states.view(b, q, self.kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v_states)
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        kv_groups = self.num_heads // self.kv_heads
        k_states = repeat_kv(k_states, kv_groups)
        v_states = repeat_kv(v_states, kv_groups)

        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights,p=0)

        attn_output = torch.matmul(attn_weights, v_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(b, q, -1)

        attn_output = self.W_output(attn_output)
        return attn_output


class LlamaDecoder(nn.Module):
    def __init__(self, config: SmolMoEConfig):
        super().__init__()
        self.self_attn = RopeAttention(config)
        self.moe = MoE(
            num_experts=config.num_experts,
            num_experts_per_token=config.num_experts_per_token,
            emb_dim=config.hidden_size,
            moe_dim=config.intermediate_size,
            noise_std=config.moe_noise_std,
        )
        self.pre_attn_rmsnorm = RMSNorm(config.hidden_size, eps=1e-5)
        self.pre_moe_rmsnorm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        residual = hidden_states
        hidden_states = self.pre_attn_rmsnorm(hidden_states)

        seq_len = attention_mask.shape[-1]
        causal_mask = torch.triu(
            torch.full(
                (seq_len, seq_len),
                fill_value=float("-inf"),
                device=hidden_states.device,
            ),
            diagonal=1,
        )

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.pre_moe_rmsnorm(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        return outputs


class SmolMoEModel(nn.Module):
    def __init__(self, config: SmolMoEConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [LlamaDecoder(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        hidden_states = self.embed_tokens(input_ids)
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        return [hidden_states]


class SmolMoELM(nn.Module):
    def __init__(self, config: SmolMoEConfig):
        super().__init__()
        self.model = SmolMoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.reset_weights_and_metrics()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return {'logits':logits}

    def get_expert_utilization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate expert utilization and load-balancing loss across layers.

        Returns
        -------
        utilization_per_layer : [L, E]
        lb_loss : scalar tensor
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        per_layer_load = []
        aux_terms = []

        for dec in self.model.layers:
            moe = dec.moe
            load = getattr(moe, "_expert_utilization", None)
            aux = getattr(moe, "_aux_lb", None)
            if load is None or aux is None:
                continue
            per_layer_load.append(load.to(device=device, dtype=dtype))
            aux_terms.append(aux.to(device))

        utilization_per_layer = (
            torch.stack(per_layer_load, dim=0) if per_layer_load else torch.empty(0)
        )
        if aux_terms:
            lb_loss = torch.stack(aux_terms).mean()
        else:
            lb_loss = torch.tensor(0.0, device=device, dtype=dtype)

        return utilization_per_layer, lb_loss.detach()

    def reset_weights_and_metrics(self) -> None:
        with torch.no_grad():
            modules = list(self.modules())[1:]
            for m in modules:
                fn = getattr(m, "reset_parameters_", None) or getattr(m, "reset_parameters", None)
                if callable(fn):
                    fn()

            for m in modules:
                if hasattr(m, "reset_parameters") or hasattr(m, "reset_parameters_"):
                    continue
                any_param = False
                for name, p in m.named_parameters(recurse=False):
                    any_param = True
                    if p.dim() == 1:
                        if name == "bias":
                            p.zero_()
                        else:
                            p.fill_(1.0)
                    else:
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
