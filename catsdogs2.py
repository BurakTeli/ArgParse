import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import time

# Veriyi ön işleme için dönüşümler
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Görüntü boyutunu ayarlama
    transforms.ToTensor(),  # Görüntüleri tensor formatına dönüştürme
])

# Eğitim veri setini yükleyin
print("Eğitim veri seti yükleniyor...")
train_dataset = datasets.ImageFolder(root='Cat_Dog_data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Test veri setini yükleyin
print("Test veri seti yükleniyor...")
test_dataset = datasets.ImageFolder(root='Cat_Dog_data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modeli önceden eğitilmiş ağırlıklarla yükleyin
print("Model yükleniyor...")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # Son katmanı iki sınıfa ayırma

# Cihaz ayarı
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Cihaz ayarlandı: {device}")
model.to(device)  # Modeli cihaza taşı

# Kayıp fonksiyonu ve optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim döngüsü
num_epochs = 55  # Eğitim epoch sayısı
for epoch in range(num_epochs):
    print(f"Eğitim Epoch {epoch + 1} başlıyor...")
    model.train()  # Modeli eğitim moduna al
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Cihaza taşı
        optimizer.zero_grad()  # Gradienleri sıfırla
        outputs = model(images)  # Modelden çıktı al
        loss = criterion(outputs, labels)  # Kayıp hesapla
        loss.backward()  # Geri yayılım
        optimizer.step()  # Ağırlıkları güncelle
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}')

    # Avg Loss 0.0100 altına inerse, eğitimi durdur
    if avg_loss < 0.0100:
        print(f"Avg Loss {avg_loss:.4f} 0.0100'ın altına indi, eğitim durduruluyor.")
        break

print("Eğitim tamamlandı.")

# Test döngüsü
model.eval()  # Modeli değerlendirme moduna al
print("Test aşaması başlıyor...")
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # Cihaza taşı
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # En yüksek olasılığa sahip sınıfı al
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
