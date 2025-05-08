import os
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image

# ==== Thiết lập cấu hình ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'data/test'
batch_size = 32
model_path = "mobilenetv2_cats.pth"

# ==== Chuẩn bị dữ liệu ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
num_classes = len(test_dataset.classes)

# ==== Khởi tạo model và load trọng số ====
model = models.mobilenet_v2(weights=None)  # Sửa 'pretrained' → 'weights'
model.classifier[1] = nn.Linear(model.last_channel, num_classes)



if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ==== Đánh giá mô hình ====
y_true, y_pred, misclassified_images = [], [], []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Đánh giá"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        # Lưu ảnh phân loại sai
        for i in range(labels.size(0)):
            if predicted[i] != labels[i]:
                misclassified_images.append((inputs[i].cpu(), labels[i].item(), predicted[i].item()))

# ==== Tính toán chỉ số đánh giá ====
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=test_dataset.classes, digits=4)
cm = confusion_matrix(y_true, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print("Classification Report:\n", report)

# ==== Vẽ Confusion Matrix ====
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ==== Phân tích lỗi ====
misclassified = cm.sum(axis=1) - np.diag(cm)
for idx, cls in enumerate(test_dataset.classes):
    print(f"Lớp '{cls}': {misclassified[idx]} lần phân loại sai")

max_mis_idx = np.argmax(misclassified)
print(f"\nLớp bị sai nhiều nhất: '{test_dataset.classes[max_mis_idx]}' ({misclassified[max_mis_idx]} lần)")

# ==== Hiển thị một số ảnh bị phân loại sai ====
print("\nHiển thị 5 ảnh bị phân loại sai đầu tiên:")

for i, (img_tensor, true_label, pred_label) in enumerate(misclassified_images[:5]):
    img = transforms.ToPILImage()(img_tensor)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f"True: {test_dataset.classes[true_label]} | Pred: {test_dataset.classes[pred_label]}")
    plt.axis('off')
    plt.show()

# ==== Kiểm tra imbalance ====
class_counts = np.bincount(y_true, minlength=num_classes)
for i, count in enumerate(class_counts):
    print(f"Lớp '{test_dataset.classes[i]}': {count} mẫu trong tập kiểm tra.")

# ==== Cảnh báo nếu số lớp model ≠ dataset ====
final_layer = model.classifier[1]

if isinstance(final_layer, nn.Sequential):
    model_output_classes = final_layer[-1].out_features  # Lấy lớp cuối cùng trong Sequential
elif isinstance(final_layer, nn.Linear):
    model_output_classes = final_layer.out_features      # Lấy trực tiếp nếu là Linear
else:
    model_output_classes = num_classes                  # fallback nếu không rõ

if model_output_classes != num_classes:
    print(f"CẢNH BÁO: Số lớp của model ({model_output_classes}) không khớp với số lớp của tập test ({num_classes})!")

