import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# Define the Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_data_path):
        self.model = self.load_model()  # Initialize the model
        self.train_data_path = train_data_path
        self.train_loader = self.load_data()

        # Set device and optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()

    def load_model(self):
        model = models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.DEFAULT')  
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model = nn.DataParallel(model)  # Wrap for multi-GPU training
        return model

    def load_data(self):
        transform_train = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_data = ImageFolder(root=self.train_data_path, transform=transform_train)
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
        return train_loader

    def get_parameters(self, config):
        if self.model is not None:
            # Access underlying model with .module
            return self.model.module.state_dict()  
        else:
            return {}

    def set_parameters(self, parameters, config):
        if self.model is not None:
            # Access underlying model with .module 
            self.model.module.load_state_dict(parameters) 

    def fit(self, parameters, config):
        if self.model is None:
            raise ValueError("Model not initialized!")

        self.model.train()
        self.model.set_parameters(parameters, config)

        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.unsqueeze(1).float()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(self.train_loader)
        return self.model.get_parameters(config), len(self.train_loader), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        if self.model is None:
            raise ValueError("Model not initialized!")

        self.model.eval()
        self.model.set_parameters(parameters, config)

        val_loss = 0.0
        correct = total = 0

        with torch.no_grad():
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1).float()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(self.train_loader)
        accuracy = correct / total * 100
        return self.model.get_parameters(config), avg_val_loss, accuracy

# Take input from the user for the folder path
train_data_path = r"./cancer/lol"

# Create multiple Flower clients with the provided folder path
num_clients = 5
clients = [FlowerClient(train_data_path) for _ in range(num_clients)]

# Start each Flower client
for client in clients:
    fl.client.start_client(
        server_address='127.0.0.1:8080',
        client=client.to_client(),
    )
