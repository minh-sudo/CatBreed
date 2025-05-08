import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# ==== Cấu hình ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'data/train'  # Thư mục train
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# ==== Chuẩn bị dữ liệu ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_classes = len(train_dataset.classes)

# ==== Load ResNet-18 pretrained ====
model = models.resnet18(pretrained=True)

# Thay lớp cuối cùng (fc) thành lớp mới phù hợp số lớp
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)

# ==== Fine-tuning ====
# Đóng băng các lớp thấp (giữ nguyên weight)
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

# ==== Định nghĩa loss và optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==== Huấn luyện ====
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

# ==== Lưu trọng số model ====
os.makedirs('weights', exist_ok=True)
torch.save(model.state_dict(), 'weights/resnet18_cats.pth')
print("Model saved to weights/resnet18_cats.pth")
