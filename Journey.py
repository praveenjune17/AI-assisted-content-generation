🚀 Updated Proof of Concept: Transformer Model with Step Duration

This new version incorporates step duration into the sequence, ensuring that both step order and time spent are considered for predicting interaction probability.


---

🔹 Step 1: Updated Data Generation

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Constants
VOCAB_SIZE = 200  # Assume max step ID is 200
PAD_TOKEN = 0
SEQ_LEN = 20  # Fixed sequence length for padding
BATCH_SIZE = 64

# Function to generate a synthetic sequence with duration
def generate_synthetic_sequence():
    seq_len = random.randint(5, SEQ_LEN - 1)  # Random length (ensuring at least 5 steps)
    sequence = [random.randint(1, 50) for _ in range(seq_len)]  # Random step IDs
    durations = [round(random.uniform(0.5, 5.0), 2) for _ in range(seq_len)]  # Random durations

    if random.random() < 0.5:  # 50% chance of interaction
        interaction_index = random.randint(2, seq_len - 1)
        sequence = sequence[:interaction_index]  # Remove 51 to prevent leakage
        durations = durations[:interaction_index]
        target = 1  # Interaction happened
    else:
        sequence.append(182)  # Stop step
        durations.append(round(random.uniform(0.5, 5.0), 2))  # Stop duration
        target = 0  # No interaction

    # Pair steps with durations
    paired_sequence = list(zip(sequence, durations))
    
    return paired_sequence, target

# Generate dataset
num_samples = 10000
dataset = [generate_synthetic_sequence() for _ in range(num_samples)]
sequences, labels = zip(*dataset)

# Padding function
def pad_sequence(seq, max_len=SEQ_LEN, pad_token=(PAD_TOKEN, 0.0)):
    return seq[:max_len] + [pad_token] * (max_len - len(seq))

padded_sequences = [pad_sequence(seq) for seq in sequences]

# Convert to tensors
X_steps = torch.tensor([[step for step, _ in seq] for seq in padded_sequences], dtype=torch.long)
X_durations = torch.tensor([[dur for _, dur in seq] for seq in padded_sequences], dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32)


---

🔹 Step 2: Updated Dataset and DataLoader

class InteractionDataset(Dataset):
    def __init__(self, X_steps, X_durations, y):
        self.X_steps = X_steps
        self.X_durations = X_durations
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_steps[idx], self.X_durations[idx], self.y[idx]

# Split data into train and test sets
train_size = int(0.8 * len(y_tensor))
train_dataset = InteractionDataset(X_steps[:train_size], X_durations[:train_size], y_tensor[:train_size])
test_dataset = InteractionDataset(X_steps[train_size:], X_durations[train_size:], y_tensor[train_size:])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


---

🔹 Step 3: Updated Transformer Model

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.positional_encoding = nn.Parameter(torch.randn(SEQ_LEN, embed_dim))

        # Duration transformation (linear projection)
        self.duration_transform = nn.Linear(1, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected output layer
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, step_ids, durations):
        # Embed step IDs
        step_embeddings = self.embedding(step_ids) + self.positional_encoding

        # Transform durations to same dimensionality as embeddings
        durations = durations.unsqueeze(-1)  # Shape: (batch, seq_len, 1)
        duration_embeddings = self.duration_transform(durations)  # Shape: (batch, seq_len, embed_dim)

        # Combine step embeddings and duration embeddings
        x = step_embeddings + duration_embeddings

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Take last token representation and classify
        x = x[:, -1, :]
        x = self.fc(x)

        return self.sigmoid(x)

# Initialize model
model = TransformerClassifier(VOCAB_SIZE)


---

🔹 Step 4: Training the Model

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train(model, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for step_ids, durations, labels in train_loader:
            step_ids, durations, labels = step_ids.to(device), durations.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(step_ids, durations).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

train(model, train_loader, optimizer, criterion)


---

🔹 Step 5: Evaluating the Model

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for step_ids, durations, labels in test_loader:
            step_ids, durations, labels = step_ids.to(device), durations.to(device), labels.to(device)
            outputs = model(step_ids, durations).squeeze()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

evaluate(model, test_loader)


---

🔹 Step 6: Extracting Contextual Embeddings

def extract_embeddings(model, step_ids, durations):
    model.eval()
    with torch.no_grad():
        step_ids, durations = step_ids.to(device), durations.to(device)
        step_embeddings = model.embedding(step_ids) + model.positional_encoding
        duration_embeddings = model.duration_transform(durations.unsqueeze(-1))
        combined_embeddings = step_embeddings + duration_embeddings
        transformed_embeddings = model.transformer_encoder(combined_embeddings)
        return transformed_embeddings.mean(dim=1)  # Aggregate embeddings

# Example: Get embeddings for first 10 sequences
sample_embeddings = extract_embeddings(model, X_steps[:10].to(device), X_durations[:10].to(device))
print(sample_embeddings.shape)  # Should be (10, embed_dim)


---

🚀 Summary of Updates

✅ Integrated duration as an extra feature
✅ Modified transformer model to process step embeddings & durations
✅ Ensured proper alignment & padding
✅ Trained & extracted contextual embeddings for downstream tasks

Would you like to add any further enhancements, such as attention visualization or hyperparameter tuning? 🚀

