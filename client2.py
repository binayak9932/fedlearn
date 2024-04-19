import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.models import efficientnet

# Define the Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_data_path):
        self.model = None
        self.train_data_path = train_data_path
        self.train_loader = self.load_data()

    def load_data(self):
        transform_train = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_data = ImageFolder(root=self.train_data_path, transform=transform_train)
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
        return train_loader

    def get_parameters(self):
        return self.model.get_parameters() if self.model else []

    def set_parameters(self, parameters):
        if self.model:
            self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        self.model.train()
        self.model.set_parameters(parameters)

        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(self.train_loader)
        return self.model.get_parameters(), len(self.train_loader), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.model.eval()
        self.model.set_parameters(parameters)

        val_loss = 0.0
        correct = total = 0

        with torch.no_grad():
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.unsqueeze(1).float()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(self.train_loader)
        accuracy = correct / total * 100
        return self.model.get_parameters(), avg_val_loss, accuracy

# Take input from the user for the folder path
train_data_path = r"./cancer/lol"

# Create a Flower client with the provided folder path
client = FlowerClient(train_data_path)

# Set device and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = efficientnet.efficientnet_b0(pretrained=True)  # Load EfficientNet with pre-trained weights
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 1)  # Replace classifier for binary classification
model = nn.DataParallel(model).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Start the Flower client with server address and the client object
fl.client.start_client(server_address="[::]:8080", client=client.to_client())