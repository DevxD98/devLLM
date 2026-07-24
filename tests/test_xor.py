from jimmylabs.autograd.nn import MLP
from jimmylabs.utils.seed import seed_everything

def test_xor_training():
    """
    Train an MLP on the 4 XOR points using the toy engine.
    Assert that after N seeded steps, loss < 0.05 and all 4 points
    are classified correctly.
    """
    seed_everything(1337)
    
    model = MLP(2, [4, 1])
    
    xs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    ys = [0.0, 1.0, 1.0, 0.0]
    
    learning_rate = 0.1
    epochs = 200
    
    # Training loop
    for _ in range(epochs):
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt)**2 for yout, ygt in zip(ypred, ys))
        
        model.zero_grad()
        loss.backward()
        
        for p in model.parameters():
            p.data -= learning_rate * p.grad
            
    # Verify loss reached near-zero
    assert loss.data < 0.05, f"Expected loss < 0.05, got {loss.data}"
    
    # Test predictions against binary classification thresholds
    preds = [model(x).data for x in xs]
    
    # [0.0, 0.0] -> 0
    assert preds[0] < 0.5
    # [0.0, 1.0] -> 1
    assert preds[1] > 0.5
    # [1.0, 0.0] -> 1
    assert preds[2] > 0.5
    # [1.0, 1.0] -> 0
    assert preds[3] < 0.5
