
# BERT-E2E-ABSA
## Nguồn gốc: https://github.com/lixin4ever/BERT-E2E-ABSA  
**BERT** cho **Phân tích cảm xúc dựa trên khía cạnh từ đầu đến cuối**

<p align="center">
    <img src="architecture.jpg" height="400"/>
</p>

---

## Yêu cầu môi trường
- Python 3.8.8  
- PyTorch 1.2.0 (đã thử nghiệm trên cả PyTorch 1.3.0)  
- ~~transformers 2.0.0~~ ➜ sử dụng **transformers 4.1.1**  
- numpy 1.16.4  
- tensorboardX 1.9  
- tqdm 4.32.1  
- Một số đoạn mã được mượn từ:
  - **allennlp** (https://github.com/allenai/allennlp)  
  - **transformers** (https://github.com/huggingface/transformers)  

---

## Kiến trúc mô hình
- **Lớp nhúng đầu vào**: BERT-Base-Uncased (12 lớp, 768 chiều ẩn, 12 đầu attention, 110 triệu tham số)  
- **Lớp đặc thù cho tác vụ**:
  - Linear  
  - Mạng nơ-ron hồi tiếp (GRU)  
  - Mạng tự chú ý (SAN, TFM)  
  - Trường ngẫu nhiên có điều kiện (CRF)  

---

## Bộ dữ liệu
- ~~Restaurant: Bộ đánh giá nhà hàng từ SemEval 2014, 2015, 2016 (rest_total)~~  
- (**QUAN TRỌNG**) Restaurant: đánh giá từ SemEval 2014 (rest14), 2015 (rest15), 2016 (rest16)  
- (**QUAN TRỌNG**) **KHÔNG** sử dụng bộ dữ liệu `rest_total`  
- Laptop: đánh giá sản phẩm laptop từ SemEval 2014 (laptop14)  

---

## Bắt đầu nhanh
### Các chiến lược gán nhãn hợp lệ:  
- **BIEOS**, **BIO**, **OT**  
Xem thêm:
- https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)  
- https://www.aclweb.org/anthology/E99-1023.pdf  
- https://www.aclweb.org/anthology/D19-5505.pdf  

### Huấn luyện lại mô hình:
```bash
python fast_run.py
```

### Huấn luyện với dữ liệu khác:
1. Đặt dữ liệu trong `./data/[TÊN_BỘ_DỮ_LIỆU]`  
2. Thiết lập `TASK_NAME` trong `train.sh`  
3. Chạy `sh train.sh`  

### Suy luận với mô hình đã huấn luyện:
1. Đặt dữ liệu trong `./data/[TÊN_BỘ_DỮ_LIỆU_EVAL]`  
2. Thiết lập `TASK_NAME` và `ABSA_HOME` trong `work.sh`  
3. Chạy `sh work.sh`  

---

## Môi trường thử nghiệm
- OS: REHL Server 6.4 (Santiago)  
- GPU: NVIDIA GTX 1080 Ti  
- CUDA: 10.0  
- cuDNN: v7.6.1  

---

## Cập nhật kết quả (**QUAN TRỌNG**)

| Mô hình | rest14 | rest15 | rest16 |
|--------|--------|--------|--------|
| E2E-ABSA (CỦA CHÚNG TÔI) | 67.10 | 57.27 | 64.31 |
| He et al. (2019) | 69.54 | 59.18 | n/a |
| Liu et al. (2020) | 68.91 | 58.37 | n/a |
| BERT-Linear (CỦA CHÚNG TÔI) | 72.61 | 60.29 | 69.67 |
| BERT-GRU (CỦA CHÚNG TÔI) | 73.17 | 59.60 | 70.21 |
| BERT-SAN (CỦA CHÚNG TÔI) | 73.68 | 59.90 | 70.51 |
| BERT-TFM (CỦA CHÚNG TÔI) | 73.98 | 60.24 | 70.25 |
| BERT-CRF (CỦA CHÚNG TÔI) | 73.17 | 60.70 | 70.37 |
| Chen và Qian (2020) | 75.42 | 66.05 | n/a |
| Liang et al. (2020) | 72.60 | 62.37 | n/a |

---

## Trích dẫn
```bibtex
@inproceedings{li-etal-2019-exploiting,
    title = "Exploiting {BERT} for End-to-End Aspect-based Sentiment Analysis",
    author = "Li, Xin  and
      Bing, Lidong  and
      Zhang, Wenxuan  and
      Lam, Wai",
    booktitle = "Proceedings of the 5th Workshop on Noisy User-generated Text (W-NUT 2019)",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-5505",
    pages = "34--41"
}
```
