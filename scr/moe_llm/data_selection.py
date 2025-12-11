# src/moe_llm/data_selection.py

from collections import defaultdict, Counter
from typing import Tuple, Dict, List, Iterable, Any

import math

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# ---------- token-level entropy helpers ----------

def per_token_entropy_from_ids(token_ids: List[int]) -> float:
    """
    Compute Shannon entropy (in bits) over token IDs in a single sequence.

    This is per-token entropy: higher values correspond to richer / less repetitive
    token distributions for that sample.
    """
    n = len(token_ids)
    if n == 0:
        return 0.0
    counts = Counter(token_ids)
    probs = [c / n for c in counts.values()]
    H = -sum(p * math.log2(p) for p in probs)
    return H


def text_per_token_entropy(text: str, tok: PreTrainedTokenizerBase) -> float:
    if not text:
        return 0.0
    ids = tok.encode(text, add_special_tokens=False)
    return per_token_entropy_from_ids(ids)


def weighted_score(
    text: str,
    prompt: str,
    tok: PreTrainedTokenizerBase,
    w_text: float,
    w_prompt: float,
) -> float:
    """
    Weighted combination of per-token entropies for the main text and its prompt.
    """
    return (
        w_text * text_per_token_entropy(text, tok)
        + w_prompt * text_per_token_entropy(prompt, tok)
    )


# ---------- stratified fixed-total selection ----------

def stratified_fixed_total_token_entropy(
    dataset: Dataset,
    total_samples: int,
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M",
    group_cols: Tuple[str, str, str] = ("audience", "format", "seed_data"),
    w_text: float = 0.8,
    w_prompt: float = 0.2,
    min_text_chars: int = 50,
    min_prompt_chars: int = 10,
    even_strategy: str = "even_then_fill",
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Select exactly `total_samples` examples (if supply allows), with:

      - Stratification by (audience, format, seed_data)
      - Within each stratum, ranking examples by a weighted token-entropy score
      - Fixed total sample budget, with quotas per group.

    Strategy:
      1. Bucket by group_cols.
      2. Filter out very short text/prompt.
      3. Within each bucket, sort by descending entropy-based score.
      4. Compute quotas:
         - "even_then_fill": distribute as evenly as possible, then assign remainder
           to groups with more available examples.
         - "proportional": quotas ~ available/eligible_total.
      5. Reallocate shortfalls in groups with insufficient supply.
      6. Optionally trim excess to hit the exact total_samples.

    Returns
    -------
    selected_indices : list[int]
        Indices into the original dataset.
    report : dict
        Diagnostics (group counts, quotas, filtering stats).
    """
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    buckets: Dict[Tuple, List[Tuple[float, int]]] = defaultdict(list)
    total_seen = 0
    total_filtered_short = 0

    for i, ex in enumerate(dataset):
        total_seen += 1
        txt = (ex.get("text") or "").strip()
        prm = (ex.get("prompt") or "").strip()
        if len(txt) < min_text_chars or len(prm) < min_prompt_chars:
            total_filtered_short += 1
            continue
        key = tuple(ex.get(col, None) for col in group_cols)
        s = weighted_score(txt, prm, tok, w_text, w_prompt)
        buckets[key].append((s, i))

    # sort each group by descending score
    for g in buckets:
        buckets[g].sort(key=lambda x: x[0], reverse=True)

    groups = list(buckets.keys())
    G = len(groups)
    eligible_total = sum(len(v) for v in buckets.values())
    if eligible_total == 0:
        return [], {
            "reason": "no_eligible_items_after_filters",
            "total_seen": total_seen,
            "total_filtered_short": total_filtered_short,
        }

    # Not enough supply: return all
    if eligible_total <= total_samples:
        selected = [idx for g in groups for _, idx in buckets[g]]
        return selected, {
            "note": "not_enough_supply; returned all eligible",
            "requested_total": total_samples,
            "selected_total": len(selected),
            "groups_total": G,
            "eligible_total": eligible_total,
        }

    # ---- Quota computation ----
    quotas = {g: 0 for g in groups}

    if even_strategy == "proportional":
        for g in groups:
            quotas[g] = round(len(buckets[g]) / eligible_total * total_samples)
    else:
        base = total_samples // G
        remainder = total_samples % G
        for g in groups:
            quotas[g] = base
        groups_sorted_by_supply = sorted(
            groups, key=lambda k: len(buckets[k]), reverse=True
        )
        for g in groups_sorted_by_supply[:remainder]:
            quotas[g] += 1

    # Cap by availability
    current_total = 0
    for g in groups:
        quotas[g] = min(quotas[g], len(buckets[g]))
        current_total += quotas[g]

    # Reallocate shortfall
    if current_total < total_samples:
        need = total_samples - current_total
        candidates: List[Tuple[float, Tuple, int]] = []
        for g in groups:
            remaining = buckets[g][quotas[g] :]
            for s, idx in remaining:
                candidates.append((s, g, idx))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, g, idx in candidates:
            if need == 0:
                break
            quotas[g] += 1
            need -= 1
        current_total = total_samples

    # Trim excess (can happen with proportional rounding)
    if current_total > total_samples:
        selected_pool: List[Tuple[float, Tuple, int]] = []
        for g in groups:
            selected_pool.extend(
                [(buckets[g][k][0], g, buckets[g][k][1]) for k in range(quotas[g])]
            )
        selected_pool.sort(key=lambda x: x[0])  # ascending by score
        to_drop = current_total - total_samples
        drop_set = set()
        for j in range(to_drop):
            _, g, idx = selected_pool[j]
            drop_set.add((g, idx))
            quotas[g] -= 1

    # Final selection
    selected_indices: List[int] = []
    for g in groups:
        selected_indices.extend([buckets[g][k][1] for k in range(quotas[g])])

    report = {
        "requested_total": total_samples,
        "selected_total": len(selected_indices),
        "groups_total": G,
        "eligible_total": eligible_total,
        "total_seen": total_seen,
        "total_filtered_short": total_filtered_short,
        "quota_summary": {
            "per_group": {str(g): quotas[g] for g in groups},
            "min_quota": min(quotas.values()) if quotas else 0,
            "max_quota": max(quotas.values()) if quotas else 0,
        },
        "even_strategy": even_strategy,
        "weights": {"w_text": w_text, "w_prompt": w_prompt},
    }
    return selected_indices, report
