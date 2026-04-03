import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

#transformation pipeline
transform = transforms.Compose([
    transforms.Resize((28, 28)),          # Resize images to 28x28
    transforms.Grayscale(num_output_channels=1), # Convert to grayscale (1 channel)
    transforms.ToTensor()                 # Convert images to PyTorch tensors
])

training_data = datasets.ImageFolder(
    root='', #root path to training data
    transform=transform
)

test_data = datasets.ImageFolder(
    root='', #root path to test data
    transform=transform
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(28*28, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, len(categories)) # Modified to use dynamic number of classes

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        output = self.l3(x)
        return output
    
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
model = Net()

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=False)

loss_fn = nn.CrossEntropyLoss() # used for categorization
learning_rate = 1e-3
# note: optimizer is Adam: one of the best optimizers to date
# it can infer learning rate and all hyper-parameters automatically
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

#testing model
sample_num = 6 # select a random sample

with torch.no_grad():
    r = model(training_data[sample_num][0])

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