import torch
from jimmylabs.model.gpt import GPT

@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, top_p: float = None) -> torch.Tensor:
    """
    Autoregressive generation loop taking O(N) forward passes without KV cache (Naive phase).
    Args:
        model: The trained GPT model.
        idx: (B, T) tensor of context token indices.
        max_new_tokens: Number of tokens to generate.
        temperature: Softmax temperature. If close to 0, acts as greedy decoding.
        top_k: If set, only keep the top_k tokens with highest probability.
        top_p: Nucleus sampling. If set, keep tokens whose cumulative probability exceeds top_p.
    Returns:
        (B, T + max_new_tokens) tensor of appended token indices.
    """
    block_size = model.config.block_size
    
    for _ in range(max_new_tokens):
        # Truncate to block_size if necessary
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # Forward pass to get logits for the sequence
        logits, _ = model(idx_cond)
        
        # We only care about the last step's logits
        logits = logits[:, -1, :] # (B, V)
        
        # Temperature scaling
        # Avoid division by zero by handling deterministic greedy decoding explicitly
        if temperature < 1e-5:
            # Deterministic greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            continue
            
        logits = logits / temperature
        
        # Top-K sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # Everything less than the minimum value in the top-k is masked
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Top-p (Nucleus) sampling
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter the mask back to the original ordering
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            
            # Apply the mask to logits to prevent sampling
            logits[indices_to_remove] = -float('Inf')
            
            # Recompute probs with the masked logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
        # Sample the next token
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        
        # Append to the sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx
