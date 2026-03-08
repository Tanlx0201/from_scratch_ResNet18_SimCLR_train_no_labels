# ResNet18 học ảnh *không gán nhãn* (SimCLR) + tối ưu CUDA

## 1) Cấu trúc
- `resnet18/` : ResNet18 from scratch (có thêm option `return_features=True` để dùng self-supervised)
- `dataset.py` :
  - `get_dataloaders(...)` (supervised, giữ lại)
  - `get_simclr_dataloader(...)` (unlabeled/SSL)
- `ssl_simclr.py` : SimCLR + NT-Xent loss
- `train_resnet.py` :
  - `--mode simclr` (mặc định) train không gán nhãn
  - `--mode supervised` train có nhãn (như cũ)

## 2) Train SimCLR (không gán nhãn)
### A) Nếu bạn đang dùng STL10 (dataset đã ở `./data`)
```bash
python train_resnet.py --mode simclr --dataset stl10 --data-root ./data --epochs 200 --batch-size 256
```

### B) Nếu bạn có folder ảnh riêng (không cần subfolder class)
Giả sử ảnh nằm trong `./data/my_unlabeled_images/` (có thể lồng nhiều thư mục con)
```bash
python train_resnet.py --mode simclr --dataset folder --data-root ./data/my_unlabeled_images --epochs 200 --batch-size 256
```

Checkpoint sẽ lưu ở `./checkpoints/`.

## 3) Các tối ưu CUDA đang bật
- `cudnn.benchmark=True` (tăng tốc cho input size cố định)
- TF32 cho matmul/cuDNN (tăng throughput trên Ampere+)
- `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`
- Transfer `non_blocking=True`
- `channels_last` memory format
- Mixed precision AMP (có thể tắt bằng `--no-amp`)

> Lưu ý: PyTorch không cho "bật nhân/luồng" GPU theo kiểu CUDA kernel thủ công; các flag trên là cách phổ biến để GPU chạy hiệu quả nhất.

## 4) Train supervised (giữ lại)
```bash
python train_resnet.py --mode supervised --epochs 100 --batch-size 128
```
