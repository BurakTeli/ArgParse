import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import time
import argparse  # argparse modülünü ekledik

# Argümanları tanımla
parser = argparse.ArgumentParser(description='PyTorch ile kedi ve köpek sınıflandırması')
parser.add_argument('--batch_size', type=int , default=32 'help= batch boyutu')
parser.add_argument('--num_epochs', type=int , defulat=21 'help= epochs sayısı')
parser.add_argument('--learning_rate' , type=float, default=32 , help='Öğrenme oranı')
args = parser.parse_args()


transforms = transforms.Compose[(
    transforms.Resize(224, 224),
    transforms.ToTensor()
)]

print("Eğitim setini yükleme")
train_dataset = dataset.ImagerFolder(root='Cat_Dog_data/train', transform=transform)
train_loader = DataLoader(test_dataset, batch_size=32 , shuffle=True)

print("Test seti yükleme")
test_dataset = datasets.ImageFolder(root='Cat_Dog_data/test' , transform=transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print("Model yükleme")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features,2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Cihaz ayarlandı: {device}")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = args.num_epochs  # Eğitim epoch sayısını argümandan al
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