import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import datetime
import matplotlib.pyplot as plt
import os

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(16 * 7 * 7, 24)
        self.fc2 = nn.Linear(24, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def save_sample_images(dataset, num_samples=5):
    # Create directory for sample images if it doesn't exist
    os.makedirs('sample_images', exist_ok=True)
    
    # Get some random training images
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(dataloader))
    
    # Plot the images
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for idx, (img, label) in enumerate(zip(images, labels)):
        axes[idx].imshow(img.squeeze(), cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'Label: {label.item()}')
    
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'sample_images/augmented_samples_{timestamp}.png')
    plt.close()

def evaluate(model, device, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.1)
    ])
    
    # Simple transform for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.MNIST('data', train=False, download=True, transform=val_transform)
    
    # Save some sample augmented images
    print("Saving sample augmented images...")
    save_sample_images(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000)

    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
                                            steps_per_epoch=len(train_loader), 
                                            epochs=1)
    
    # Training loop
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if batch_idx % 100 == 0:
            # Check accuracy on validation set
            val_acc = evaluate(model, device, val_loader)
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}')
            model.train()

    # Final validation accuracy
    final_acc = evaluate(model, device, val_loader)
    print(f'\nFinal Validation Accuracy: {final_acc:.4f}')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'model_{timestamp}.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")
    
    return model

if __name__ == "__main__":
    train_model() 