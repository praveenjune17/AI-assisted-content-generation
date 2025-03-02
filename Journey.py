import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformer Decoder Model for Regression
class TransformerDecoderRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4):
        super(TransformerDecoderRegression, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.memory_embedding = nn.Linear(input_dim, hidden_dim)  # Ensure memory has correct shape
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)  # Regression output
        self.sigmoid = nn.Sigmoid()  # Ensure output is in range [0,1]
        
    def forward(self, x, memory):
        x = self.embedding(x)  # Embed input
        memory = self.memory_embedding(memory)  # Embed memory to same dimension
        x = self.transformer_decoder(x, memory)  # Decode sequence
        x = self.fc(x[:, -1, :])  # Use last time step
        return self.sigmoid(x)  # Regression output in [0,1]

# Synthetic Dataset Generation Functions
def generate_synthetic_sequence():
    # Generate a random sequence length between 5 and 19 (SEQ_LEN=20)
    seq_len = random.randint(5, 19)
    sequence = [random.randint(1, 50) for _ in range(seq_len)]  # Random step IDs
    durations = [round(random.uniform(0.5, 5.0), 2) for _ in range(seq_len)]  # Random durations
    # For regression, we assign a random target in [0,1]
    target = round(random.uniform(0.0, 1.0), 2)
    # Pair steps with durations
    paired_sequence = list(zip(sequence, durations))
    return paired_sequence, target

def pad_sequence(seq, max_len=20, pad_token=(0, 0.0)):
    return seq[:max_len] + [pad_token] * (max_len - len(seq))

# Custom Dataset that returns a two-tuple: (inputs, target)
class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences  # Each sequence is a list of (step, duration) pairs
        self.targets = targets      # Regression targets in [0,1]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# Generate dataset
num_samples = 10
dataset_raw = [generate_synthetic_sequence() for _ in range(num_samples)]
sequences, targets = zip(*dataset_raw)
padded_sequences = [pad_sequence(seq) for seq in sequences]

# Split dataset (for simplicity, we'll use the same dataset for train and test here)
train_dataset = SequenceDataset(padded_sequences, targets)
test_dataset = SequenceDataset(padded_sequences, targets)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Model Setup
input_dim = 2  # (Step ID, Duration)
hidden_dim = 16
model = TransformerDecoderRegression(input_dim, hidden_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training Loop wrapped in a function
def train_model(model, dataloader, optimizer, loss_fn, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            # Move data to device
            inputs = inputs.to(device)  # Shape: (batch_size, seq_len, 2)
            targets = targets.to(device)  # Shape: (batch_size)
            
            optimizer.zero_grad()
            # Create dummy memory input with the same shape as inputs
            memory = torch.zeros_like(inputs).to(device)
            outputs = model(inputs, memory)
            loss = loss_fn(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}")

# Evaluation Loop wrapped in a function
def evaluate_model(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size, seq_length, _ = inputs.shape
            memory = torch.zeros((batch_size, seq_length, inputs.shape[-1]), device=inputs.device)
            outputs = model(inputs, memory)
            loss = loss_fn(outputs.squeeze(), targets)
            total_loss += loss.item()
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss (MSE): {avg_loss:.4f}")
    return all_preds, all_targets

# Run training and evaluation
trained_model = train_model(model, train_loader, optimizer, loss_fn, num_epochs=10)
preds, target = evaluate_model(model, test_loader, loss_fn)
