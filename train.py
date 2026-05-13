import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import json
import sys
from model import CRNN

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, labels_file, characters, img_w=100, img_h=32, min_len=4, max_len=4):
        self.data_dir = data_dir
        self.img_w = img_w
        self.img_h = img_h
        self.characters = characters
        self.char_map = {char: i + 1 for i, char in enumerate(characters)}
        self.min_len = min_len
        self.max_len = max_len

        self.data = []
        with open(labels_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        img_path = os.path.join(self.data_dir, "images", record['filename'])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Create a black image if missing
            img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.img_w, self.img_h))

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # [1, h, w]

        # Only use verified labels that fit format
        is_correct = record.get('isCorrect', False)
        label_raw = record.get('label', '')

        if is_correct and len(label_raw.strip()) > 0:
            label_text = label_raw
        else:
            # We don't want to train on unverified labels
            return torch.from_numpy(img), torch.LongTensor([]), 0

        # Clean label: only supported characters and no whitespace
        label_text = "".join([c for c in label_text if c in self.char_map and not c.isspace()])

        # If it's a hallucination (too long) or empty, skip
        if len(label_text) > self.max_len or len(label_text) < self.min_len:
            return torch.from_numpy(img), torch.LongTensor([]), 0

        target = [self.char_map[c] for c in label_text]
        target_len = len(target)

        return torch.from_numpy(img), torch.LongTensor(target), target_len

def collate_fn(batch):
    imgs, targets, target_lens = zip(*batch)
    imgs = torch.stack(imgs)
    # Concatenate targets for CTCLoss (it can take either a padded tensor or a single concatenated tensor)
    targets_flat = torch.cat(targets)
    target_lens = torch.IntTensor(target_lens)
    return imgs, targets_flat, target_lens

def train(num_epochs=100, min_len=4, max_len=4):
    # Config
    DATA_DIR = "datasets/ocr"
    LABELS_FILE = os.path.join(DATA_DIR, "labels.jsonl")

    print(f"Using labels from: {LABELS_FILE}")

    characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    n_class = len(characters) + 1  # +1 for blank

    dataset = CaptchaDataset(DATA_DIR, LABELS_FILE, characters, min_len=min_len, max_len=max_len)
    if len(dataset) == 0:
        print("No data to train on.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = CRNN(32, 1, n_class, 256)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training on {len(dataset)} samples for {num_epochs} epochs...")
    for epoch in range(num_epochs): # More epochs for better results
        model.train()
        epoch_loss = 0
        for i, (imgs, targets, target_lens) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward
            preds = model(imgs)  # [w, b, n_class]
            preds_size = torch.IntTensor([preds.size(0)] * preds.size(1))

            loss = criterion(preds.log_softmax(2), targets, preds_size, target_lens)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader)}")

    # Export to ONNX
    print("Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 1, 32, 100)
    onnx_path = os.path.join(DATA_DIR, "ocr_model.onnx")

    # Define common export parameters
    modern_params = {
        'input_names': ['input'],
        'output_names': ['output'],
        'dynamic_shapes': {'input': {0: torch.export.Dim("batch_size")}},
        'opset_version': 18
    }
    legacy_params = {
        'input_names': ['input'],
        'output_names': ['output'],
        'dynamic_axes': {'input': {0: 'batch_size'}, 'output': {1: 'batch_size'}},
        'opset_version': 18
    }

    try:
        # Attempt modern export (default in recent PyTorch versions, may require onnxscript)
        torch.onnx.export(model, dummy_input, onnx_path, **modern_params)
        
        # Verify opset version as version conversion might fail silently
        import onnx
        exported_model = onnx.load(onnx_path)
        actual_opset = exported_model.opset_import[0].version
        if actual_opset != 18:
            raise RuntimeError(f"Expected opset 18 but got {actual_opset}")

        print(f"Model exported to {onnx_path} with opset {actual_opset}")
    except (ImportError, ModuleNotFoundError) as e:
        if "onnxscript" in str(e):
            print("Warning: onnxscript not found. Falling back to legacy export with dynamo=False...")
            try:
                # Explicitly set dynamo=False to avoid onnxscript requirement
                torch.onnx.export(model, dummy_input, onnx_path, dynamo=False, **legacy_params)
                print(f"Model exported to {onnx_path} using legacy exporter.")
            except Exception as le:
                print(f"Error: Legacy export also failed: {le}")
                print("Please install onnxscript: pip install onnxscript")
        else:
            print(f"Export failed with unexpected import error: {e}")
    except Exception as e:
        print(f"Modern export failed: {e}. Attempting legacy export...")
        try:
            torch.onnx.export(model, dummy_input, onnx_path, dynamo=False, **legacy_params)
            print(f"Model exported to {onnx_path} using legacy exporter.")
        except Exception as le:
            print(f"Error: Legacy export also failed: {le}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=4)
    args = parser.parse_args()

    if args.min_len > args.max_len:
        print("Error: --min-len cannot be greater than --max-len")
        sys.exit(1)

    train(num_epochs=args.epochs, min_len=args.min_len, max_len=args.max_len)
