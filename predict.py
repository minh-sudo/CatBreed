import torch
import os
from torchvision import models, transforms
from PIL import Image

# THIET BI CHAY
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# LOAD MO HINH
model = models.mobilenet_v2()
model.classifier[1]=torch.nn.Linear(model.last_channel,12) #12 GIONG MEO
model.load_state_dict(torch.load("mobilenetv2_cats.pth",map_location=device))
model.to(device)
model.eval()
class_name = ['abyssinian','americanshorthair','bengal','mainecoon','mumbai','persian','ragdoll','scottishfold','siamese','sphinx','tortoiseshell','tuxedo']

# TIEN XU LY
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# LOAD ANH
results = []
test_folder = "D:/python/bt_ai/data/test"
for root, _, files in os.walk(test_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, filename)
            image = Image.open(img_path).convert("RGB")  
            input_tensor = transform(image).unsqueeze(0).to(device)

            # DU DOAN
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
            # DUONG DAN
            rel_path = os.path.relpath(img_path, start=os.getcwd())
            results.append((rel_path.replace('\\', '/'), class_name[predicted.item()]))

# TẠO FILE HTML
with open("results.html", "w", encoding="utf-8") as f:
    f.write("<html><body><h1>Kết quả dự đoán</h1>\n")
    for img_path, label in results:
        f.write(f"<div style='margin-bottom:20px;'>\n")
        f.write(f"<img src='{img_path}' height='200'><br>\n")
        f.write(f"<b>Dự đoán:</b> {label}<br>\n")
        f.write("</div>\n")
    f.write("</body></html>")





