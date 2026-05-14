import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
import sys
import logging
from model import CRNN

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, labels_file, characters, img_w=100, img_h=32, min_len=4, max_len=4, data=None):
        self.data_dir = data_dir
        self.img_w = img_w
        self.img_h = img_h
        self.characters = characters
        self.char_map = {char: i + 1 for i, char in enumerate(characters)}
        self.min_len = min_len
        self.max_len = max_len

        if data is not None:
            self.data = data
        else:
            self.data = []
            with open(labels_file, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    # Pre-filter for valid records
                    is_correct = record.get('isCorrect', False)
                    label_raw = record.get('label', '')
                    if is_correct and len(label_raw.strip()) > 0:
                        label_text = "".join([c for c in label_raw if c in self.char_map and not c.isspace()])
                        if self.min_len <= len(label_text) <= self.max_len:
                            record['clean_label'] = label_text
                            self.data.append(record)

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

        label_text = record['clean_label']
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

def train(num_epochs=50, min_len=4, max_len=4):
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

    # Split data records
    all_data = dataset.data
    np.random.seed(42)
    np.random.shuffle(all_data)
    
    train_split = int(0.8 * len(all_data))
    train_records = all_data[:train_split]
    val_records = all_data[train_split:]

    # # Oversample lowercase in training
    # lowercase_records = [r for r in train_records if any(c.islower() for c in r['clean_label'])]
    # # Double the lowercase records to give them more weight
    # train_records_oversampled = train_records + lowercase_records
    
    train_dataset = CaptchaDataset(DATA_DIR, LABELS_FILE, characters, min_len=min_len, max_len=max_len, data=train_records)
    val_dataset = CaptchaDataset(DATA_DIR, LABELS_FILE, characters, min_len=min_len, max_len=max_len, data=val_records)

    dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = CRNN(32, 1, n_class, 256).to(device)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

    # Early stopping config
    best_val_loss = float('inf')
    patience = 100 # Increased patience
    counter = 0

    logging.info(f"Starting training on {len(train_dataset)} training samples and {len(val_dataset)} validation samples for {num_epochs} epochs...")
    for epoch in range(num_epochs): 
        model.train()
        train_loss = 0
        for i, (imgs, targets, target_lens) in enumerate(dataloader_train):
            imgs = imgs.to(device)
            targets = targets.to(device)
            # target_lens and preds_size should stay on CPU for CTCLoss
            
            # If target_lens is 0, skip this batch element
            if target_lens.sum() == 0:
                continue
            
            optimizer.zero_grad()

            # Forward
            preds = model(imgs)  # [w, b, n_class]
            preds_size = torch.IntTensor([preds.size(0)] * preds.size(1))

            # Use log_softmax for CTC loss
            log_probs = preds.log_softmax(2)
            loss = criterion(log_probs, targets, preds_size, target_lens)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN or Inf loss detected at epoch {epoch}, batch {i}")
                continue

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (imgs, targets, target_lens) in enumerate(dataloader_val):
                imgs = imgs.to(device)
                targets = targets.to(device)
                # target_lens should stay on CPU

                if target_lens.sum() == 0:
                    continue
                
                preds = model(imgs)
                preds_size = torch.IntTensor([preds.size(0)] * preds.size(1))
                loss = criterion(preds.log_softmax(2), targets, preds_size, target_lens)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(dataloader_train) if len(dataloader_train) > 0 else 0
        avg_val_loss = val_loss / len(dataloader_val) if len(dataloader_val) > 0 else 0

        scheduler.step(avg_val_loss)
        logging.info(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, LR: {optimizer.param_groups[0]['lr']}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_model.pth"))
            logging.info(f"Saved best model at epoch {epoch}")
        else:
            counter += 1
            if counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Export to ONNX
    print("Exporting to ONNX...")
    model.eval()
    model.to('cpu') # Move to CPU for export
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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=4)
    args = parser.parse_args()

    if args.min_len > args.max_len:
        print("Error: --min-len cannot be greater than --max-len")
        sys.exit(1)

    train(num_epochs=args.epochs, min_len=args.min_len, max_len=args.max_len)
