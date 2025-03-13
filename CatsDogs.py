import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Boyutlandırma
    transforms.ToTensor(),          # Tensöre dönüştürme
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizasyon
])

# Veri setlerinin yüklenmesi
train_dataset = datasets.ImageFolder(root='/Users/buraktelli/Desktop/Ordulu_proj/Cat_Dog_data/train', transform=transform)
test_dataset = datasets.ImageFolder(root='/Users/buraktelli/Desktop/Ordulu_proj/Cat_Dog_data/test', transform=transform)

# Veri seti boyutlarını yazdır
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# DataLoader'ların oluşturulması
train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False, pin_memory=True)

# Modelinizi burada tanımlayın
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    #bacth
    #con
    nn.Flatten(),
    nn.Linear(16 * 112 * 112, 2)  # 2 sınıf için
).to(device)  # Modeli GPU'ya taşı

# Kayıp fonksiyonu ve optimizasyon tanımları
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Model eğitim döngüsü
num_epochs = 55
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} başlıyor...")
    model.train()  # Modeli eğitim moduna al
    train_loss = 0.0
    correct = 0
    total = 0

    # Eğitim döngüsü
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Verileri GPU'ya taşı
        optimizer.zero_grad()  # Gradienten sıfırlama
        outputs = model(images)  # Modelden çıkış
        loss = criterion(outputs, labels)  # Kayıp hesaplama
        loss.backward()  # Backpropagation
        optimizer.step()  # Ağırlıkları güncelleme

        # Kayıp ve doğruluk hesaplama
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1} tamamlandı. Kayıp: {train_loss / len(train_loader):.4f}, Doğruluk: {train_accuracy:.2f}%")

    # Test döngüsü
    model.eval()  # Modeli değerlendirme moduna al
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Verileri GPU'ya taşı
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Kayıp: {test_loss / len(test_loader):.4f}, Test Doğruluk: {test_accuracy:.2f}%")

# Modeli kaydetme
torch.save(model.state_dict(), "cat_dog_model.pth")
print("Model kaydedildi.")
