from dataclasses import dataclass
import yaml

@dataclass
class GPTConfig:
    """
    Configuration for the JimmyLabs GPT model.
    Follows constraints and fields defined in SPEC.md §9.
    """
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float = 0.1
    weight_tying: bool = True
    
    def __post_init__(self):
        # Validate that n_embd is cleanly divisible by n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
            
    @classmethod
    def from_yaml(cls, path: str) -> 'GPTConfig':
        """Load configuration from a YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        # Extract only the keys relevant to GPTConfig
        # Allowing YAML files to contain training hyperparams without crashing
        model_keys = ['vocab_size', 'n_layer', 'n_head', 'n_embd', 'block_size', 'dropout', 'weight_tying']
        config_data = {k: v for k, v in data.items() if k in model_keys}
        
        return cls(**config_data)
