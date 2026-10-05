# Lab Day 2 - DeepWeeds (Bùi Đăng Khoa, 2A202602617)

## Chạy lại
- Môi trường: Google Colab, GPU Tesla T4. Thư viện: torch, timm, fvcore, scikit-learn, pandas, openpyxl, scipy (ghi version bằng `pip freeze` khi chạy lại).
- Notebook Colab:
  - **Phần 1** (B01-B05, T01-T06): [Colab Notebook Part 1](https://colab.research.google.com/drive/1Z8nAwoQRkHTj6hNjn8T0QI16EQN4rAds?usp=sharing)
  - **Phần 2** (T00 seed 1-2, F01 seed 0-2, suy luận, F02, đánh giá: [Colab Notebook Part 2](https://colab.research.google.com/drive/1sRFT61_x5g4Fhw7P90Xv2Ny-8EBtZCWq?usp=sharing)
- Thứ tự: (1) tải images.zip, kiểm MD5 b7b30f96d466fba86016aa5a26606e0f, giải nén vào /content/data, tải 4 file CSV fold 0;
  (2) `dataset.check_split`; (3) B01-B05 (`train.run`); (4) T01-T06; (5) T00 seed 1,2; (6) F01 seed 0,1,2 với `save_test_predictions=True`;
  (7) dự đoán test mốc T00 từ best.pt; (8) temperature scaling -> F02; (9) `eval.py score/grade`.
- Seed: 0, 1, 2. Mọi lần chạy dùng `train.run(Config(...))`.

## Ghi chú
- `eval.py` dùng bản gốc, không sửa.
- `curves/` đặt tên `<exp_id>_seed<k>_<backbone>.png`.
- T00 seed 0 chính là B03 (copy kết quả sang runs/T00).
- F01 = ConvNeXt-T + CutMix (dự đoán chưa hiệu chuẩn); F02 = F01 + temperature scaling (T khớp trên val).
- Không commit checkpoint (best.pt) và dữ liệu ảnh.
