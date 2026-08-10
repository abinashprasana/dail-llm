from pathlib import Path

import pytest

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
