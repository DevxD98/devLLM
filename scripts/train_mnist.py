import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from jimmylabs.utils.seed import seed_everything
import os

class MNISTMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_config(path="configs/mnist.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_mnist(config_path="configs/mnist.yaml", fast_dev_run=False):
    # Setup
    seed_everything(42)
    config = load_config(config_path)
    device = get_device()
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Use a local cache for the dataset
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_dataset = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)

    if fast_dev_run:
        # Use a tiny subset for the smoke test
        train_dataset = torch.utils.data.Subset(train_dataset, range(64))
        test_dataset = torch.utils.data.Subset(test_dataset, range(64))
        config['epochs'] = 5

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Model
    model = MNISTMLP(config['hidden_size']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training Loop
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if fast_dev_run and batch_idx == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

    # Testing
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return accuracy

if __name__ == "__main__":
    train_mnist()
