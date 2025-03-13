import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import argparse
import matplotlib.pyplot as plt

# Argümanları tanımla
parser = argparse.ArgumentParser(description='PyTorch ile kedi ve köpek sınıflandırması')
parser.add_argument('--batch_size', type=int, default=32, help='Batch boyutu')
parser.add_argument('--num_epochs', type=int, default=21, help='Epoch sayısı')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Öğrenme oranı')
parser.add_argument('--model_type', type=str, choices=['resnet18', 'custom'], default='resnet18', help='Kullanılacak model tipi (resnet18 veya custom)')
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print("Eğitim setini yükleme")
train_dataset = datasets.ImageFolder(root='Cat_Dog_data_2/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

print("Test setini yükleme")
test_dataset = datasets.ImageFolder(root='Cat_Dog_data_2/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print("Model yükleme")
if args.model_type == 'resnet18':
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
else:
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Cihaz ayarlandı: {device}")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Eğitim ve test aşamaları için kayıplar ve doğruluklar
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

num_epochs = args.num_epochs
for epoch in range(num_epochs):
    print(f"Eğitim Epoch {epoch + 1} başlıyor...")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch + 1} tamamlandı. Kayıp: {train_loss:.4f}, Doğruluk: {train_accuracy:.2f}%")

    if train_loss < 0.0100:
        print(f"Avg Loss {train_loss:.4f} 0.0100'ın altına indi, eğitim durduruluyor.")
        break

print("Eğitim tamamlandı.")

# Test aşaması
print("Test başlıyor...")
model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct / total
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)
print(f"Test Kayıp: {test_loss:.4f}, Test Doğruluk: {test_accuracy:.2f}%")

# Eğitim ve doğruluk grafiği oluşturma
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 6))

# Kayıp grafiği
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Eğitim Kayıp', color='blue')
plt.axhline(y=test_losses[0], color='orange', linestyle='--', label='Test Kayıp')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.title('Eğitim ve Test Kayıp Grafiği')
plt.legend()

# Doğruluk grafiği
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Eğitim Doğruluk', color='green')
plt.axhline(y=test_accuracies[0], color='red', linestyle='--', label='Test Doğruluk')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk (%)')
plt.title('Eğitim ve Test Doğruluk Grafiği')
plt.legend()

plt.tight_layout()
plt.show()
