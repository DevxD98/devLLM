import torch
from jimmylabs.model.gpt import GPT

@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, top_p: float = None, use_cache: bool = False) -> torch.Tensor:
    """
    Autoregressive generation loop taking O(N) forward passes without KV cache (Naive phase)
    or O(T) with KV cache.
    """
    block_size = model.config.block_size
    past_key_values = None
    
    for _ in range(max_new_tokens):
        if use_cache and past_key_values is not None:
            # Trim cache if we are at max block_size
            seq_len = past_key_values[0][0].size(-2)
            if seq_len >= block_size:
                # Keep the last block_size - 1 tokens to leave room for the 1 new token
                trim_len = seq_len - (block_size - 1)
                past_key_values = [(k[:, :, trim_len:], v[:, :, trim_len:]) for k, v in past_key_values]
            
            idx_cond = idx[:, -1:]
        else:
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # Forward pass to get logits for the sequence
        if use_cache:
            logits, _, past_key_values = model(idx_cond, use_cache=True, past_key_values=past_key_values)
        else:
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
