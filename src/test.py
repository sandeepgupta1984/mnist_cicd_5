import torch
import torch.nn as nn
from torchvision import datasets, transforms
from train import MNIST_CNN
from torch.optim.lr_scheduler import OneCycleLR

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = MNIST_CNN().to(device)
    
    # Test 1: Parameter count
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    assert total_params < 25000, f"Model has too many parameters: {total_params}"
    print("✓ Parameter count test passed")

    # Test 2: Input shape compatibility
    test_input = torch.randn(1, 1, 28, 28).to(device)
    try:
        output = model(test_input)
        print("✓ Input shape (28x28) test passed")
    except Exception as e:
        raise AssertionError(f"Failed to process 28x28 input: {e}")

    # Test 3: Output shape
    assert output.shape == (1, 10), f"Wrong output shape: {output.shape}"
    print("✓ Output shape test passed")

    # Test 4: Quick training and accuracy check
    transform = transforms.Compose([
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
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    # Calculate number of steps for OneCycleLR
    steps_per_epoch = 100  # Number of training iterations
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=steps_per_epoch, epochs=1)
    
    # Train for multiple iterations on the same batch
    model.train()
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    
    # Training loop with learning rate scheduling
    print("\nTraining Progress:")
    for step in range(steps_per_epoch):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (step + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                output = model(data)
                pred = output.argmax(dim=1)
                correct = pred.eq(target).sum().item()
                accuracy = correct / len(target)
                print(f"Step {step + 1}, Accuracy: {accuracy:.4f}, Loss: {loss.item():.4f}")
            model.train()

    # Final accuracy check
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        accuracy = correct / len(target)
        print(f"\nFinal accuracy: {accuracy:.4f}")
        assert accuracy > 0.95, f"Accuracy {accuracy:.4f} is below threshold (0.95)"
        print("✓ Accuracy test passed")

if __name__ == "__main__":
    test_model() 