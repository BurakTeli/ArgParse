import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse

# Argparse ile komut satırı argümanlarını alma
parser = argparse.ArgumentParser(description="Kedi ve köpek sınıflandırma modeli")
parser.add_argument('--train_path', type=str, required=True, help='Eğitim veri seti yolu')
parser.add_argument('--test_path', type=str, required=True, help='Test veri seti yolu')
parser.add_argument('--batch_size', type=int, default=32, help='Batch boyutu')
parser.add_argument('--num_epochs', type=int, default=5, help='Epoch sayısı')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Öğrenme oranı')
args = parser.parse_args()

# Veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Veri setlerinin yüklenmesi
train_dataset = datasets.ImageFolder(root=args.train_path, transform=transform)
test_dataset = datasets.ImageFolder(root=args.test_path, transform=transform)

# Veri seti boyutlarını yazdır
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# DataLoader'ların oluşturulması
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Modelinizi burada tanımlayın
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 112 * 112, 2)  # 2 sınıf için
)

# Kayıp fonksiyonu ve optimizasyon tanımları
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Model eğitim döngüsü
for epoch in range(args.num_epochs):
    print(f"Epoch {epoch + 1} başlıyor...")
    model.train()  # Modeli eğitim moduna al

    for images, labels in train_loader:
        optimizer.zero_grad()  # Gradienten sıfırlama
        outputs = model(images)  # Modelden çıkış
        loss = criterion(outputs, labels)  # Kayıp hesaplama
        loss.backward()  # Backpropagation
        optimizer.step()  # Ağırlıkları güncelleme

    print(f"Epoch {epoch + 1} tamamlandı. Kayıp: {loss.item()}")

print("Eğitim tamamlandı.")
