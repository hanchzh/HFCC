import torch
from transformers import LlamaForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader


class IoTDevice:
    def __init__(self, device_id: int, device_type: str, learning_rate: float, model_name: str, local_data: Dataset):
        self.device_id = device_id  # Unique identifier for the device
        self.device_type = device_type  # Type of the device (e.g., low_power, edge_node)
        self.learning_rate = learning_rate  # Learning rate for the device
        self.model = LlamaForSequenceClassification.from_pretrained(model_name)  # Load model
        self.local_data = local_data  # Local dataset for training
        self.local_epochs = 1  # Number of training epochs
        self.batch_size = 64  # Local batch size
        self.dataloader = DataLoader(local_data, batch_size=self.batch_size, shuffle=True)  # DataLoader for batching

    def train(self):
        """Train the model on local data."""
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)  # Initialize optimizer
        self.model.train()  # Set model to training mode

        for epoch in range(self.local_epochs):
            for batch in self.dataloader:
                inputs, labels = batch  # Unpack the inputs and labels from the batch
                optimizer.zero_grad()  # Reset gradients
                outputs = self.model(inputs)  # Forward pass
                loss = self.compute_loss(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

            print(f"Device {self.device_id} | Epoch {epoch + 1}/{self.local_epochs} | Loss: {loss.item():.4f}")

    def compute_loss(self, outputs, labels):
        """Compute the loss for the outputs."""
        return torch.nn.CrossEntropyLoss()(outputs, labels)  # Return CrossEntropy loss

    def get_local_parameters(self):
        """Get the current local model parameters."""
        return self.model.state_dict()  # Return the state dict of the model

    def load_local_parameters(self, local_params):
        """Load local parameters into the model."""
        self.model.load_state_dict(local_params)  # Load parameters into model

    def save_model(self, output_dir: str):
        """Save the local model to the specified output directory."""
        save_path = f"{output_dir}/device_{self.device_id}_model.bin"
        torch.save(self.model.state_dict(), save_path)  # Save model parameters
        print(f"Model saved for Device {self.device_id} at {save_path}")

    # Example of a custom dataset class for demonstration


class CustomDataset(Dataset):
    def __init__(self, data):  # Initialization with data
        self.data = data  # Store data

    def __len__(self):
        return len(self.data)  # Return length of the dataset

    def __getitem__(self, idx):
        # Example of unpacking data
        item = self.data[idx]
        inputs = torch.tensor(item['input_ids'])  # Convert input IDs to tensor
        labels = torch.tensor(item['labels'])  # Convert labels to tensor
        return inputs, labels  # Return input and label as a tuple