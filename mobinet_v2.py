import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2 , MobileNet_V2_Weights


# 1. THIET BI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# 2. TIEN XU LY DU LIEU

transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# 3. LOAD DU LIEU TU THU MUC 'data'
data_dir = 'D:/python/bt_ai/data'
batch_size = 32

datasets = {
    x: datasets.ImageFolder(root=f'{data_dir}/{x}', transform=transform[x])
    for x in ['train', 'val']
}
dataloaders = {
    x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)
    for x in ['train', 'val']
}
class_names = datasets['train'].classes
num_classes = len(class_names)


# 4. KHOI TAO MOBILENETV2
# weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = mobilenet_v2(weights=None) 

model.load_state_dict(torch.load("D:/python/bt_ai/mobilenet_v2-b0353104.pth"))
model.eval()

# Freeze phần feature nếu muốn huấn luyện nhanh
# for param in model.features.parameters():
#     param.requires_grad = False

# Tùy chỉnh phần classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, num_classes)
)
model = model.to(device)

# 5. LOSS VA OPTIMIZER

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)


# 6. HAM TRAIN

def train_model(model, dataloaders, criterion, optimizer, num_epochs=12):
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)

    return train_loss_history, val_loss_history


# 7. MAIN
if __name__ == '__main__':
    train_loss, val_loss = train_model(model, dataloaders, criterion, optimizer, num_epochs=12)


# 8. LUU VAO 'mobilenetv2_cats.pth'

torch.save(model.state_dict(), 'mobilenetv2_cats.pth')
print(f"Đã lưu mô hình vào {os.path.abspath('mobilenetv2_cats.pth')}")
