1.Sử dụng kiến trúc: Đề xuất Mobilenet_v2
+   Nhanh, nhẹ, tốn ít tài nguyên, chính xác với dataset ~5000 ảnh

2.Triển khai bằng PyTorch

3.Huấn luyện mô hình
+   Theo dõi được: loss và accuracy trung bình sau mỗi forder phân loại
+   Xử lý được overfitting: Dropout(0.3)
+   Lưu ra file mobilenet_cats.pth

4.Thử dự đoán với test + lưu kết quả ra web
+ lưu ra web result.html

LƯU Ý:
+  Thay đổi đường dẫn file mục 3,4.
+  Dùng bản PyTorch có CUDA 11.8:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
