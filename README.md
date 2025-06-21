# ChestXray-CFT
Deeplearning Model for ChestXray14 Classification

# Phân Loại và Phân Đoạn Bệnh Lý X-Quang Ngực

Dự án này triển khai một pipeline học sâu để phân loại và phân đoạn các bệnh lý trong hình ảnh X-quang ngực. Hệ thống hỗ trợ phân loại đa nhãn và phân đoạn cho năm bệnh lý: **Tăng kích thước tim (Cardiomegaly)**, **Tràn dịch màng phổi (Effusion)**, **Xẹp phổi (Atelectasis)**, **Tràn khí màng phổi (Pneumothorax)**, và **Thâm nhiễm (Infiltration)**. Pipeline sử dụng **EfficientNet-B3** làm backbone, **Feature Pyramid Network (FPN)** để trích xuất đặc trưng, và **Mini-Transformer** để cải thiện hiệu quả phân loại. Mô hình phân đoạn tạo ra các mặt nạ bệnh lý được dẫn hướng bởi heatmap từ mô hình phân loại.

## Cấu trúc dự án

- **src/**: Thư mục chứa mã nguồn chính
  - `dataset.py`: Định nghĩa `ChestXrayDataset` cho phân loại và `ChestXraySegDataset` cho phân đoạn, bao gồm tăng cường dữ liệu và tiền xử lý.
  - `model.py`: Triển khai `ChestXrayModel` cho phân loại và `ChestXraySegModel` cho phân đoạn, sử dụng EfficientNet-B3, FPN, và Mini-Transformer.
  - `train.py`: Script huấn luyện mô hình phân loại, sử dụng Focal Loss và tích lũy gradient.
  - `seg_train.py`: Script huấn luyện mô hình phân đoạn, kết hợp BCE Loss và Dice Loss.
  - `utils.py`: Các hàm tiện ích bao gồm Focal Loss, Dice Loss, phân chia dữ liệu theo tầng (stratified split), và các chỉ số đánh giá (AUC, Dice, IoU).
- **main.py**: Điểm bắt đầu để huấn luyện mô hình phân loại.
- **seg_main.py**: Điểm bắt đầu để huấn luyện mô hình phân đoạn.
- **test.py**: Đánh giá mô hình phân loại trên tập val hoặc test, so sánh AUC với các benchmark.
- **seg_test.py**: Đánh giá mô hình phân đoạn, tính toán Dice, IoU, độ chính xác pixel, và lưu kết quả trực quan.
- **configs/config.yaml**: File cấu hình cho tham số mô hình, dữ liệu, và huấn luyện.
- **data/**: Thư mục lưu các file CSV chia dữ liệu và dữ liệu hộp giới hạn (`BBox_List_2017.csv`).
- **models/**: Thư mục lưu trọng số mô hình đã huấn luyện.
- **results/**: Thư mục lưu kết quả trực quan của phân đoạn.

## Tính năng

- **Phân loại**: Phân loại đa nhãn sử dụng Focal Loss, tích hợp metadata (tuổi, giới tính, góc chụp), và tạo heatmap để định vị bệnh lý.
- **Phân đoạn**: Tạo mặt nạ cho từng bệnh lý dựa trên chú thích hộp giới hạn, được dẫn hướng bởi heatmap từ mô hình phân loại.
- **Xử lý dữ liệu**: Phân chia dữ liệu theo tầng dựa trên bệnh lý và ID bệnh nhân để đảm bảo phân phối cân bằng giữa các tập train/val/test.
- **Đánh giá**:
  - Phân loại: Tính AUC, độ chính xác, và F1 score, so sánh với benchmark từ Table 17 trong tài liệu `ARXIV_V5_CHESTXRAY.pdf`.
  - Phân đoạn: Tính Dice, IoU, và độ chính xác pixel.
- **Huấn luyện**: Hỗ trợ dừng sớm (early stopping), lập lịch tốc độ học cosine annealing, và tích lũy gradient để tối ưu bộ nhớ.
- **Trực quan hóa**: Lưu kết quả phân đoạn dưới dạng hình ảnh với heatmap màu và chú thích hộp giới hạn.

## Yêu cầu

- Python 3.8 trở lên
- PyTorch
- torchvision
- pandas
- numpy
- scikit-learn
- timm
- pyyaml
- tqdm
- PIL (Pillow)

Cài đặt các thư viện:
```bash
pip install torch torchvision pandas numpy scikit-learn timm pyyaml tqdm pillow
```

## Dữ liệu

- **Dữ liệu đầu vào**:
  - Hình ảnh X-quang ngực trong thư mục `config['data']['image_dir']`.
  - Metadata và nhãn trong file Excel được chỉ định tại `config['data']['data_file']`.
  - Chú thích hộp giới hạn trong file `data/BBox_List_2017.csv`.
- **Tiền xử lý**:
  - Hình ảnh được resize về 224x224, chuẩn hóa với mean/std của ImageNet.
  - Tăng cường dữ liệu: lật ngẫu nhiên, xoay, thay đổi màu sắc, làm mờ Gaussian, và xóa ngẫu nhiên.
  - Metadata được chuẩn hóa: tuổi giới hạn trong khoảng 0-100, mã hóa giới tính và góc chụp.

## Cấu hình chính

- **Mô hình**:
  - `name`: `chestxray_model` (phân loại) hoặc `chestxray_seg_model` (phân đoạn).
  - `num_classes`: 5 (cho năm bệnh lý).
  - `pretrained`: Sử dụng EfficientNet-B3 đã được huấn luyện trước trên ImageNet.
- **Dữ liệu**:
  - `data_file`: Đường dẫn đến file Excel chứa nhãn và metadata.
  - `image_dir`: Đường dẫn đến thư mục chứa hình ảnh X-quang.
  - `train_split`, `val_split`, `test_split`: Tỷ lệ chia dữ liệu (ví dụ: 0.7, 0.15, 0.15).
- **Huấn luyện**:
  - `batch_size`: Kích thước batch (ví dụ: 16).
  - `num_epochs`: Số epoch tối đa (ví dụ: 50).
  - `patience`: Độ kiên nhẫn cho dừng sớm (ví dụ: 5).
  - `learning_rate`: Tốc độ học ban đầu (ví dụ: 0.0001).

## Kết quả mẫu

Mô hình phân loại được đánh giá trên tập test với các chỉ số sau (ngày 20/05/2025):

| Bệnh lý               | AUC        | Accuracy   | F1 Score   | 
|-----------------------|------------|------------|------------|
| Tăng kích thước tim   | 0.8769     | 0.9418     | 0.4200     | 
| Tràn dịch màng phổi   | 0.8781     | 0.8344     | 0.6488     |
| Xẹp phổi              | 0.8042     | 0.7848     | 0.5131     | 
| Tràn khí màng phổi    | 0.8698     | 0.9287     | 0.4828     |
| Thâm nhiễm            | 0.7160     | 0.6883     | 0.5370     | 
| **Trung bình**        | **0.8290** | **0.8356** | **0.5203** |

## Lưu ý

- Đảm bảo file `BBox_List_2017.csv` có các cột: `Image Index`, `Finding Label`, `Bbox [x`, `y`, `w`, `h]`.
- Mô hình phân loại cần được huấn luyện trước để tạo heatmap cho phân đoạn.
- Trực quan hóa trong `seg_test.py` giả định có font `arial.ttf` để vẽ text; nếu không có, sẽ dùng font mặc định.

