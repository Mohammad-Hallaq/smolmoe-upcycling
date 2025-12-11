# SmolMoE Upcycling: From Dense LM to Mixture-of-Experts

A lightweight implementation of Mixture-of-Experts (MoE) language models with sparse upcycling capabilities. This project demonstrates how to convert pre-trained dense language models into efficient MoE architectures and provides a complete training pipeline with advanced dataset selection strategies.

## Overview

This repository provides four key capabilities:

1. **SmolMoE Architecture** - A compact MoE implementation with rotary attention, top-k routed feed-forward blocks, and expert utilization diagnostics
2. **Sparse Upcycling** - Convert pre-trained dense Hugging Face causal LMs into MoE models by intelligently mapping FFN weights into expert banks
3. **Continued Pretraining** - Train MoE routers on the `cosmopedia-100k` dataset with comprehensive LM and MoE-specific metrics
4. **Entropy-Based Dataset Selection** - Optimize training efficiency using token-entropy stratified sampling with fixed budgets

## Installation

```bash
git clone https://github.com/Mohammad-Hallaq/smolmoe-upcycling.git
cd smolmoe-upcycling

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

## Project Structure

```
src/moe_llm/
├── config.py           # Configuration and HF adapter
├── models.py           # Core MoE architecture
├── utils.py            # Utilities and helpers
├── metrics.py          # Generic metric plotting
├── generation.py       # Text generation utilities
├── checkpoints.py      # Pre-trained weight management
├── upcycling.py        # Dense-to-MoE conversion
├── data.py             # Dataset loading and processing
├── losses.py           # Loss functions
├── training.py         # Training loop
├── moe_metrics.py      # MoE-specific metrics
└── data_selection.py   # Entropy-based sampling

experiments/
├── run_sanity_generation.py
├── run_upcycling_demo.py
├── run_continued_pretraining.py
├── run_continued_pretraining_entropy_selection.py
└── demo_entropy_scores.py

tests/
├── test_expert_load_balance.py
└── test_upcycling_equivalence.py
```

## Quick Start

### 1. Sanity Check with Pre-trained SmolMoE

Test the pre-trained model with simple text generation:

```bash
python -m experiments.run_sanity_generation
```

This downloads pre-trained weights from Hugging Face, loads them into SmolMoELM, and runs greedy generation on a test prompt.

### 2. Sparse Upcycling: Convert Dense Model to MoE

Transform a dense Hugging Face model into an MoE architecture:

```bash
python -m experiments.run_upcycling_demo \
  --checkpoint <your_hf_dense_model>
```

**What this does:**
- Loads your dense causal LM and tokenizer
- Builds a compatible SmolMoELM architecture
- Demonstrates the difference between random initialization and upcycling
- Applies `upcycle_from_dense` to map FFN weights into expert banks
- Validates that the upcycled MoE reproduces the dense model's outputs

This implements the "Sparse Upcycling" technique: repurposing a dense checkpoint's FFN weights into multiple specialized experts.

### 3. Continued Pretraining

Train the MoE router on the cosmopedia dataset:

```bash
python -m experiments.run_continued_pretraining \
  --checkpoint <your_hf_dense_model>
```

**Training features:**
- Uses `HuggingFaceTB/cosmopedia-100k` corpus
- Block size: 256 tokens
- Training budget: 1,000 examples
- Optimized with Adafactor and inverse-sqrt learning rate schedule
- Gradient accumulation for stability

**Metrics tracked:**
- Train/eval cross-entropy loss
- Load-balancing loss (Switch-Transformer style)
- Expert balance score
- Balanced perplexity: `exp(CE) × exp(β(1-B))` where B is normalized expert balance

### 4. Entropy-Based Dataset Selection

Improve training efficiency with smart data selection:

```bash
python -m experiments.run_continued_pretraining_entropy_selection \
  --checkpoint <your_hf_dense_model>
```

**Selection strategy:**
- Processes entire `cosmopedia-100k` dataset
- Groups samples by (audience, format, seed_data)
- Ranks examples using token-level entropy scores
- Allocates fixed budget (1,000 samples) strategically across groups
- Prioritizes informative, non-repetitive content

Run `python -m experiments.demo_entropy_scores` to see entropy scoring on toy examples.

## MoE-Specific Metrics

### Expert Balance Score
Normalized entropy measuring expert utilization across layers:
- **1.0** = perfectly balanced usage
- **Low values** = one or few experts dominating

### Balanced Perplexity
Custom metric combining language modeling quality with expert diversity:

```
Balanced PPL = exp(CE) × exp(β(1-B))
```

This penalizes models that achieve low perplexity by collapsing to a few experts, encouraging both strong LM performance and healthy expert diversity.

## Testing

Run the test suite:

```bash
pytest -q
```

**Included tests:**
- `test_expert_load_balance.py` - Validates load-balancing loss computation
- `test_upcycling_equivalence.py` - Verifies generation equivalence after upcycling

## Limitations

- Designed for small-to-medium scale experimentation, not production deployments
- Upcycling assumes LLaMA-like architecture and naming conventions
- Router training uses simplified single-stage continued pretraining
- Limited to causal language modeling tasks

## Future Work

- Temperature-based and stochastic routing mechanisms
- Alternative router training schedules and auxiliary losses
- Expert specialization for multi-task datasets (code, math, chat)
- Scaling to larger model architectures
- Dynamic expert capacity and routing strategies

## Citation & Inspiration

This project draws inspiration from:

- **Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints** - Core upcycling methodology
- **Switch Transformer** - Load balancing approach

Please refer to the original papers for detailed methodology and theoretical foundations.

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

This code originated from a take-home challenge and has been refactored into a standalone research project for the community.