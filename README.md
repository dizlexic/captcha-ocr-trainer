# OCR Model Training & Inference

This project provides tools to train a CRNN (Convolutional Recurrent Neural Network) OCR model and test it using ONNX runtime.

## Overview
- Architecture: CNN (feature extraction) + BiLSTM (sequence modeling) + CTC Loss.
- Training: PyTorch-based training script.
- Inference: ONNX Runtime support for efficient deployment.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+

### Python Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

---

## 2. Data Preparation

Create your dataset in the `datasets/ocr/` directory:
- `datasets/ocr/images/`: Place your captcha images here.
- `datasets/ocr/labels.jsonl`: Create a file where each line is a JSON object containing the filename, label, and verification status:
  ```json
  {"filename": "image1.png", "label": "ABCD", "isCorrect": true}
  ```

---

## 3. Model Training

The training script reads the labels from `labels.jsonl`, trains the CRNN model, and exports it to ONNX.

1. Start training:
   ```bash
   python train.py
   ```
2. The script will:
   - Load the dataset from `datasets/ocr/`.
   - Train for a configured number of epochs.
   - Export the final model to `datasets/ocr/ocr_model.onnx` (and `ocr_model.onnx.data`).

---

## 4. Inference & Testing

Use `test_onnx.py` to test the model:

- Test a single image:
  ```bash
  python test_onnx.py --image path/to/image.png
  ```
- Test a dataset directory and compare against labels:
  ```bash
  python test_onnx.py --dataset_dir datasets/ocr
  ```

---

## Troubleshooting

### ONNX Export Errors
Ensure that `img_h` is a multiple of 16 (default 32) and the training dataset includes enough samples to avoid empty label issues.
