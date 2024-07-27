from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import spectral
from sklearn.model_selection import train_test_split


class HyperspectralDataset1():
    def __init__(self):
        self.data = loadmat('Indian_pines_corrected.mat')
    
    def __getitem__(self, q):
        sample = self.data[q]
        return torch.tensor(sample, dtype=torch.float32)

class HyperspectralDataset2():
    def __init__(self):
        self.labels = loadmat('Indian_pines.mat')
    
    def __getitem__(self, q):
        label = self.labels[q]
        return torch.tensor(label, dtype=torch.long)

class modelDefination(nn.Module):
    def __init__(self, n_input_features):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, n_input_features)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


train_dataset1 = HyperspectralDataset1
train_dataset2 = HyperspectralDataset2

X_train, X_test, = train_test_split(train_dataset1 , test_size=0.2, random_state=1)
y_train, y_test = train_test_split(train_dataset2, test_size=0.2, random_state=1)
train_loader = DataLoader(X_train, batch_size=32, shuffle=True)


num_classes = len(train_dataset1.shape())
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')


train_model(model, train_loader, criterion, optimizer, num_epochs=100)

def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')


evaluate_model(model, test_loader)




