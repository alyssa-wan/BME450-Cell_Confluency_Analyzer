import torch
import os
from torch import nn
from PIL import Image
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models as models

augment = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(200),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
base_dir = '/Users/alyssawan/Documents/BME450 Final Project/BME 450 Project Images/Train'
test_dir = '/Users/alyssawan/Documents/BME450 Final Project/BME 450 Project Images/Test'

#delete all augmented files first to avoid duplicates if the code is run multiple times
#for class_name in ['Low Confluency', 'Medium Confluency', 'High Confluency']:
#    class_dir = os.path.join(base_dir, class_name)
#    deleted = 0
#    for filename in os.listdir(class_dir):
#        if filename.startswith('aug_'):
#            os.remove(os.path.join(class_dir, filename))
#            deleted += 1
#    print(f"{class_name}: deleted {deleted} augmented files, {len(os.listdir(class_dir))} originals remaining")

#Augmenting the training data by applying random transformations to each image and saving the augmented images back to the same directory with a new filename. This will help increase the diversity of the training data and improve model generalization.
#for class_name, num_copies in [('Low Confluency', 2), ('Medium Confluency', 2), ('High Confluency', 2)]:
#    class_dir = os.path.join(base_dir, class_name)
#    for filename in os.listdir(class_dir):
#        if filename.endswith(('.PNG', '.JPG', '.JPEG', '.png', '.jpg', '.jpeg')):
#            img = Image.open(os.path.join(class_dir, filename))
#            for i in range(num_copies):
#               augmented = augment(img)
#               save_image(augmented, os.path.join(class_dir, f'aug_{i}_{filename}'))
#    print(f"{class_name}: {len(os.listdir(class_dir))} total files after augmentation")

#transformation pipeline (Updated to include data augmentation techniques)
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(200),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

#No augmentation on test data, only resizing and normalization
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
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

#Switched from an CNN architecture to pretrained ResNet for better performance on image data
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    
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
    model.eval()
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
    
# This was added to allow "Progressive Resizing" (switching from 128 to 224)
def get_data_loaders(img_size, batch_size):
    train_trans = transforms.Compose([
        # NEW: RandomResizedCrop is better than CenterCrop; it simulates different 
        # views of the well, making the model more robust to cell placement.
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # NEW: Cells have no orientation; flips help
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        # REVISED: ImageNet-specific normalization for better use of pretrained weights
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    test_trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    train_ds = datasets.ImageFolder(root=base_dir, transform=train_trans)
    test_ds = datasets.ImageFolder(root=test_dir, transform=test_trans)

    counts = [train_ds.targets.count(i) for i in range(len(train_ds.classes))]
    weights = [1.0 / counts[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return DataLoader(train_ds, batch_size=batch_size, sampler=sampler), \
           DataLoader(test_ds, batch_size=batch_size, shuffle=False)
#training the model
model = Net(num_classes=len(categories))

batch_size = 16 # #smaller batch size can help with generalization, but may increase training time. Adjust based on dataset size and hardware capabilities.
# To address class imbalance, we can use a WeightedRandomSampler to give more weight to underrepresented classes during training.
counts = [training_data.targets.count(i) for i in range(len(categories))]
weights = [1.0 / counts[t] for t in training_data.targets]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=sampler)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=False)
        
loss_fn = nn.CrossEntropyLoss() # used for categorization
learning_rate = 1e-5
# note: optimizer is Adam: one of the best optimizers to date
# it can infer learning rate and all hyper-parameters automatically
optimizer = optim.Adam([
    {'params': model.model.conv1.parameters(), 'lr': 5e-6},
    {'params': model.model.layer1.parameters(), 'lr': 5e-6},
    {'params': model.model.layer2.parameters(), 'lr': 5e-6},
    {'params': model.model.layer3.parameters(), 'lr': 2e-5},
    {'params': model.model.layer4.parameters(), 'lr': 2e-5},
    {'params': model.model.fc.parameters(), 'lr': 1e-5}
], lr=5e-5, weight_decay=1e-3)

epochs = 40
best_acc = 0
patience, wait = 10, 0
save_path = '/Users/alyssawan/Documents/BME450 Final Project/best_model.pt'
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.2)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    correct = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    scheduler.step(correct)
    
    if correct > best_acc:
        best_acc = correct
        torch.save(model.state_dict(), save_path)
        print(f"  New best accuracy: {100*best_acc:.1f}% — model saved")
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {t+1}, best: {100*best_acc:.1f}%")
            break

model.load_state_dict(torch.load(save_path, weights_only=True))
print("Done!")

print(training_data.classes)
print([training_data.targets.count(i) for i in range(len(categories))])

# ← Add confusion matrix code here, after training finishes -- Testing data sets
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

all_preds, all_labels = [], []

with torch.no_grad():
    for X, y in test_dataloader:
        pred = model(X)
        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=categories))
# End of confusion matrix code

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

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