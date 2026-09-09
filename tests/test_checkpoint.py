from pathlib import Path

import pytest
import torch

from dail_llm.api.service import ModelService, PromptValidationError
from dail_llm.config import CKPT_PATH
from dail_llm.inference import ModelWrapper


@pytest.mark.integration
def test_checkpoint_loads_and_generates():
    if not Path(CKPT_PATH).exists():
        pytest.skip("Checkpoint is not present")
    wrapper = ModelWrapper(device="cpu")
    text = wrapper.generate("The Minister for", max_new_tokens=2, temperature=0.8)
    assert text.startswith("The Minister for")
    assert len(text) == len("The Minister for") + 2
    with torch.random.fork_rng():
        torch.manual_seed(123)
        first = wrapper.generate("The Minister for", max_new_tokens=8, temperature=0.8)
        torch.manual_seed(123)
        second = wrapper.generate("The Minister for", max_new_tokens=8, temperature=0.8)
    assert first == second
    service = ModelService.__new__(ModelService)
    service.wrapper = wrapper
    attention = service.attention("The Minister for", layer=3, head=None)
    matrix = torch.tensor(attention["matrices"])
    assert matrix.shape == (8, 16, 16)
    assert torch.allclose(matrix.sum(-1), torch.ones(8, 16), atol=1e-6)
    assert torch.count_nonzero(torch.triu(matrix, diagonal=1)) == 0
    assert set(attention) == {
        "prompt",
        "labels",
        "layer",
        "head",
        "matrices",
        "filtered_characters",
    }
    with pytest.raises(PromptValidationError):
        service.attention("The Minister for", layer=99, head=0)
