import pytest
from jimmylabs.model.config import GPTConfig
import os

def test_config_loads_baseline_yaml():
    """
    Test that the config can be successfully loaded from the v0.1 baseline YAML
    and contains the expected values from SPEC.md.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'model_v0_1_char_100k.yaml'
    )
    
    config = GPTConfig.from_yaml(config_path)
    
    assert config.vocab_size == 65
    assert config.n_layer == 4
    assert config.n_head == 4
    assert config.n_embd == 128
    assert config.block_size == 128
    assert config.dropout == 0.1
    assert config.weight_tying is True

def test_config_validates_head_divisibility():
    """
    Test that GPTConfig rejects configurations where n_embd is not divisible by n_head.
    """
    with pytest.raises(ValueError, match="must be divisible"):
        GPTConfig(
            vocab_size=65,
            n_layer=4,
            n_head=3,      # 128 % 3 != 0
            n_embd=128,
            block_size=128
        )
