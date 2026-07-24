from jimmylabs.autograd.nn import MLP
from jimmylabs.utils.seed import seed_everything

def train_xor():
    """
    Train an MLP on the 4 XOR points using the toy micrograd-style engine.
    """
    # Determinism
    seed_everything(1337)
    
    # 2 inputs, 1 hidden layer with 4 neurons, 1 output
    model = MLP(2, [4, 1])
    
    # XOR truth table
    xs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    ys = [0.0, 1.0, 1.0, 0.0]
    
    epochs = 200
    learning_rate = 0.1
    
    print("Training XOR...")
    for k in range(epochs):
        # Forward pass
        ypred = [model(x) for x in xs]
        
        # Mean squared error loss
        loss = sum((yout - ygt)**2 for yout, ygt in zip(ypred, ys))
        
        # Zero gradients
        model.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update weights (gradient descent)
        for p in model.parameters():
            p.data -= learning_rate * p.grad
            
        if k % 50 == 0:
            print(f"Step {k:3d} | Loss: {loss.data:.4f}")

    print(f"\nFinal Loss: {loss.data:.4f}")
    print("Predictions:")
    for x, yout in zip(xs, ypred):
        print(f"  {x} -> {yout.data:.4f}")

if __name__ == "__main__":
    train_xor()
