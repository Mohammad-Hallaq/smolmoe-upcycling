# src/moe_llm/checkpoints.py
from pathlib import Path
from typing import Union

from huggingface_hub import hf_hub_download


def download_smolmoe_weights(
    repo_id: str = "dsouzadaniel/C4AI_SmolMoELM",
    filename: str = "trial_weights.pt",
    local_dir: Union[str, Path] = ".",
) -> str:
    """
    Download pre-trained weights for the SmolMoE model from Hugging Face Hub.

    Returns
    -------
    path : str
        Local filesystem path to the downloaded weights file.
    """
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )
    return path
