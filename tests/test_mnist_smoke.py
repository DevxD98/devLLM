import torch
import torch.nn as nn
import torch.optim as optim
from scripts.train_mnist import MNISTMLP, get_device
from jimmylabs.utils.seed import seed_everything

def test_mnist_overfit_a_batch():
    """
    Overfit-a-batch integration test.
    Trains a small MLP on a single fixed batch of dummy MNIST data
    on the CPU to ensure the graph is wired correctly and the model can memorize.
    """
    seed_everything(1337)
    
    # Use CPU for fast CI style tests
    device = torch.device('cpu')
    
    # 1. Create a tiny fixed batch
    batch_size = 10
    # MNIST images are 1x28x28
    dummy_data = torch.randn(batch_size, 1, 28, 28, device=device)
    # 10 classes
    dummy_targets = torch.randint(0, 10, (batch_size,), device=device)
    
    # 2. Instantiate model and optimizer
    model = MNISTMLP(hidden_size=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 3. Train on the single batch for a few iterations
    epochs = 50
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(dummy_data)
        loss = criterion(output, dummy_targets)
        loss.backward()
        optimizer.step()
        
    # 4. Check if loss is near zero
    assert loss.item() < 0.05, f"Model failed to overfit the batch. Final loss: {loss.item()}"
    
    # 5. Check accuracy on the overfitted batch
    model.eval()
    with torch.no_grad():
        output = model(dummy_data)
        preds = output.argmax(dim=1)
        correct = (preds == dummy_targets).sum().item()
        
    assert correct == batch_size, f"Model only got {correct}/{batch_size} correct on overfitted batch"
