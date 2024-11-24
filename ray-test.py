import ray
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        return F.relu(self.fc1(x))

# Define the training function
def train_fn():
    # Setup data
    data = torch.randn(100, 10)
    target = torch.randn(100, 10)
    dataset = TensorDataset(data, target)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Setup model and optimizer
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(10):
        model.train()
        for batch_data, batch_target in train_loader:
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}, Loss {loss.item()}")

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Create a Trainer
trainer = TorchTrainer(train_fn)

# Run distributed training
trainer.fit()

# Shutdown Ray
ray.shutdown()