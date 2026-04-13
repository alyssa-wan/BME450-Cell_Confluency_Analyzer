import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

#transformation pipeline (Updated to include data augmentation techniques)
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(200),          # ← crops out the black microscope border
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#No augmentation on test data, only resizing and normalization
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(200),          # ← same crop on test
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


training_data = datasets.ImageFolder(
    root='/Users/alyssawan/Documents/BME450 Final Project/BME 450 Project Images/Train', #root path to training data
    transform=transform_train
)

test_data = datasets.ImageFolder(
    root='/Users/alyssawan/Documents/BME450 Final Project/BME 450 Project Images/Test', #root path to test data
    transform=transform_test   
)

#printing out data
categories = training_data.classes

# select a random sample from the training set
sample_num = 0
# print(training_data[sample_num])
print('Inputs sample - image size:', training_data[sample_num][0].shape)
print('Label:', training_data[sample_num][1], '\n')

import matplotlib.pyplot as plt

ima = training_data[sample_num][0]
print('Inputs sample - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
ima = (ima - ima.mean())/ ima.std()
print('Inputs sample normalized - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
iman = ima.permute(1, 2, 0) # needed to be able to plot
plt.imshow(iman)

#Switched to an CNN architecture for better performance on image data
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),           # → 112x112

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),           # → 56x56

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),           # → 28x28

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),           # → 14x14
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # → [batch, 256, 1, 1]

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),           # reduces overfitting
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#testing the model
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
#training the model
model = Net(num_classes=len(categories))

batch_size = 32 #smaller batch size can help with generalization, but may increase training time. Adjust based on dataset size and hardware capabilities.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=False)

loss_fn = nn.CrossEntropyLoss() # used for categorization
learning_rate = 1e-3
# note: optimizer is Adam: one of the best optimizers to date
# it can infer learning rate and all hyper-parameters automatically
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

#testing model
sample_num = 6 # select a random sample

with torch.no_grad():
    r = model(training_data[sample_num][0].unsqueeze(0))

print('neural network output pseudo-probabilities:', r)
print('neural network output class number:', torch.argmax(r).item())
print('neural network output, predicted class:', categories[torch.argmax(r).item()])

#test results
import random
import matplotlib.pyplot as plt
import torch # Added import statement for torch

num_display = 3 # Number of test images to display

fig, axes = plt.subplots(1, num_display, figsize=(15, 5))
axes = axes.flatten()

# Get unique random indices for display
sample_indices = random.sample(range(len(test_data)), min(num_display, len(test_data)))

with torch.no_grad():
    for i, sample_idx in enumerate(sample_indices):
        image, true_label_idx = test_data[sample_idx]

        # Add a batch dimension for the model input
        image_input = image.unsqueeze(0)

        # Get model prediction
        output = model(image_input)
        predicted_label_idx = torch.argmax(output).item()

        # Get true and predicted class names
        true_label_name = categories[true_label_idx]
        predicted_label_name = categories[predicted_label_idx]

        # Determine if prediction is correct
        is_correct = (predicted_label_idx == true_label_idx)
        color = 'green' if is_correct else 'red'

        # Prepare image for display
        # Denormalize if normalization was applied (example: if scaled to mean=0, std=1)
        # For simplicity, assuming ToTensor() put values in [0,1], no further denorm needed for display
        # If the image was normalized with mean/std, inverse transform would be needed.
        # E.g., image = image * std + mean
        display_image = image.squeeze(0).cpu().numpy() # Removed .permute(1, 2, 0)

        # Display the image
        axes[i].imshow(display_image, cmap='gray')
        axes[i].set_title(f"True: {true_label_name}\nPred: {predicted_label_name}", color=color)
        axes[i].axis('off')

plt.tight_layout()
plt.show()