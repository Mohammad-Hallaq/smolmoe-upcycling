
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_llm.config import from_hf_causal_lm_config
from moe_llm.models import SmolMoELM
from moe_llm.upcycling import upcycle_from_dense


def greedy_ids(model, tokenizer, prompt, max_new_tokens=32, device="cpu"):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    ids = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(**inputs)
        next_id = torch.argmax(out["logits"][0, -1]).item()
        ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([[next_id]], device=device)],
            dim=1,
        )
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return ids


def test_upcycled_moe_matches_dense(checkpoint="...", max_new_tokens=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()
    tok = AutoTokenizer.from_pretrained(checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = from_hf_causal_lm_config(dense.config)
    moe = SmolMoELM(cfg).to(device).eval()

    upcycle_from_dense(dense, moe, layers="all")

    prompt = "Where is the Great Wall?"
    dense_ids = greedy_ids(dense, tok, prompt, max_new_tokens, device=device)
    moe_ids = greedy_ids(moe, tok, prompt, max_new_tokens, device=device)

    assert dense_ids == moe_ids, "Upcycled MoE does not reproduce dense generations"