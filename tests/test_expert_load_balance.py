# tests/test_expert_load_balance.py

import torch

from moe_llm.config import SmolMoEConfig
from moe_llm.models import SmolMoELM
from moe_llm.checkpoints import download_smolmoe_weights


def test_load_balancer_loss_matches_reference(atol: float = 1e-2):
    """
    Sanity-check that the MoE load-balancing loss matches the reference value
    for the provided pre-trained checkpoint.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = SmolMoEConfig()
    model = SmolMoELM(config).to(device)

    weights_path = download_smolmoe_weights(
        repo_id="dsouzadaniel/C4AI_SmolMoELM",
        filename="trial_weights.pt",
        local_dir=".",
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)

    # Forward at least once to populate expert utilization buffers.
    # If the checkpoint was saved after some forward passes, you may not need this;
    # but it's safer to run a dummy pass.
    dummy_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, 8),  # [batch, seq_len]
        device=device,
    )
    dummy_mask = torch.ones_like(dummy_input, device=device)
    _ = model(dummy_input, dummy_mask)

    # Reference value from the assignment
    correct_lb_loss = torch.tensor(1.0, device=device)

    _, lb_loss = model.get_expert_utilization()
    print(f"(Expected) Load Balance Loss => {correct_lb_loss:0.2f}")
    print(f"(Actual)   Load Balance Loss => {lb_loss:0.2f}")

    assert torch.isclose(lb_loss, correct_lb_loss, atol=atol), \
        "Load-balance loss does not match the reference value."
