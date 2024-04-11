
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch
import time
totalTimeStart = time.time()
from data_process import train_data, test_data
import torch
import torch.nn as nn
from torchvision import models

from transformers import ViTConfig, ViTForImageClassification
import torch.nn.functional as F

test_data_length = len(train_data)


# Assuming EEG data is provided in train_data and test_data as a list of (EEG_signal, label) tuples
# from data_process import train_data, test_data

# Custom dataset class for EEG data
class EEGDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_signal, label = self.data[idx]
        eeg_signal = eeg_signal.reshape(1, eeg_signal.shape[0], 1)  # Reshape to [1, length, 1] to mimic [C, H, W] format

        if self.transform:
            eeg_signal = self.transform(eeg_signal)

        label = torch.tensor(label, dtype=torch.long)  # Assuming labels are class indices
        return eeg_signal, label

# EEGClassifier model definition

class EEGClassifier(nn.Module):
    def __init__(self, num_classes=3, final_layer_size=32):
        super(EEGClassifier, self).__init__()
        self.final_layer_size = final_layer_size
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1, 512, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, final_layer_size, kernel_size=(3, 3), stride=2),  # Additional layer
            nn.ReLU(),
            # Add more layers as necessary
        )
        self.reduce_channels = nn.Conv2d(final_layer_size, 3, kernel_size=(1, 1))  # Reduce to 3 channels to fit EfficientNet input

        # Load a pre-trained EfficientNet model
        self.efficient_net = models.efficientnet_b0(pretrained=True)
        # Replace the classifier with a new one for your specific number of classes
        num_features = self.efficient_net.classifier[1].in_features  # Get the input feature size of the original classifier
        self.efficient_net.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.deconv(x)
        x = self.reduce_channels(x)
        # Ensure the input size matches what EfficientNet expects, typically 224x224
        x = F.interpolate(x, size=(self.final_layer_size, self.final_layer_size), mode='bilinear', align_corners=False)
        logits = self.efficient_net(x)
        return logits

# Now you can specify the input size when you create the model instance

model = EEGClassifier()
model = model.float()  # Convert model to float

# Load data
# Assuming `train_data` is available and properly formatted
train_dataset = EEGDataset(train_data, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop


print("\n\n\nNumber of test samples:", test_data_length, "\n")
print("Starting Training\n")
batchesPerEpoch = int(test_data_length/32)
print("Batches per Epoch: ",  batchesPerEpoch)

totalTrainingTimeStart = time.time()
epochs = 3
numberOfDisplays = 6
batchesPerDisplay = int(batchesPerEpoch/numberOfDisplays)
if(batchesPerDisplay == 0):
    batchesPerDisplay = 1
print("Number of Epochs: ", epochs,"\n")
for epoch in range(epochs):
    total_correct = 0
    total_samples = 0
    runsThisEpoch = 0
    trainingTimeThisEpochStart = time.time()
    
    for data, labels in train_loader:
        
        data = data.float()  # Convert data to float precision

        # Forward pass
        outputs = model(data)
        


        class_indices = labels.max(dim=1)[1]
        loss = criterion(outputs, class_indices)


        _, predicted = outputs.max(1)  # Get the indices of the max log-probability
        total_correct += (predicted == class_indices).sum().item()
        total_samples += labels.size(0)

        # Backward and optimize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Extra Information

        runsThisEpoch += 1
        if (runsThisEpoch % batchesPerDisplay == 0):
            print("Batches this Epoch: ", runsThisEpoch)
      
    accuracy = 100 * total_correct / total_samples
    print(f'\nEpoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
    print("Time taken for this Epoch: ", time.time() - trainingTimeThisEpochStart,'\n')

print("\nTotal Training Time: ", time.time() - totalTrainingTimeStart)


# Assuming `test_data` is available and properly formatted
test_dataset = EEGDataset(test_data, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set model to evaluation mode
model.eval()

# Initialize variables to track test accuracy
total_correct = 0
total_samples = 0

# Disable gradient computation for evaluation to save memory and computations
testingTimeStart = time.time()
testingRunBatchesCount = 0
print("\n\nStarting Testing\n")


with torch.no_grad():
    for data, labels in test_loader:
        data = data.float()  # Ensure data is the correct type

        # Forward pass
        outputs = model(data)
        testingRunBatchesCount += 1
        if(testingRunBatchesCount % 500 == 0):
            print("Data tested: ", testingRunBatchesCount*32)
        
        # Get predictions and update test accuracy
        _, predicted = outputs.max(1)
        class_indices = labels.max(dim=1)[1]  # Assuming labels are one-hot encoded
        total_correct += (predicted == class_indices).sum().item()
        total_samples += labels.size(0)

        if(testingRunBatchesCount == 10000):# Will never break if over 2000
            print("\nBreaking out of loop\n")
            break

# Calculate and print test accuracy
test_accuracy = 100 * total_correct / total_samples
print(f'Tested {total_samples} samples\nTest Accuracy: {test_accuracy:.2f}%, Time taken for testing: {time.time() - testingTimeStart:.2f}s')
print("\nFull Runtime: ", time.time() - totalTimeStart, "\n")