from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataset(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert PIL Image to Tensor
        # transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=15)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=15)

    return train_loader, test_loader

