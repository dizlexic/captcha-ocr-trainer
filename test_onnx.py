import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
import json

# Character mapping must match the training script
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# train.py uses 1-based indexing, with 0 as blank
CHAR_MAP = {i + 1: char for i, char in enumerate(CHARACTERS)}

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Resize to match training inputs
    img = cv2.resize(img, (100, 32)) # img_w=100, img_h=32
    img = img.astype(np.float32) / 255.0
    # Add channel dimension: [h, w] -> [1, h, w]
    img = np.expand_dims(img, axis=0)
    # Add batch dimension: [1, h, w] -> [1, 1, h, w]
    img = np.expand_dims(img, axis=0)
    return img

def ctc_greedy_decode(preds):
    # preds shape: [sequence_length, batch_size, n_class]
    # Take argmax along the class dimension
    preds_argmax = np.argmax(preds, axis=2) # [seq_len, batch_size]
    
    # Greedy decode: merge blanks (0) and consecutive same characters
    decoded = []
    prev_char_idx = None
    
    # Process the first batch item (assuming batch_size=1)
    for char_idx in preds_argmax[:, 0]:
        if char_idx != 0: # 0 is blank
            if char_idx != prev_char_idx:
                decoded.append(CHAR_MAP.get(char_idx, ''))
        prev_char_idx = char_idx
    return "".join(decoded)

def run_inference(model_path, image_path, ground_truth=None):
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return None

    img = preprocess_image(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})[0]
    
    result = ctc_greedy_decode(output)
    
    msg = f"File: {os.path.basename(image_path)} | Predicted: {result}"
    if ground_truth is not None:
        status = "Correct" if result == ground_truth else "Incorrect"
        msg += f" | {status}"
        if status == "Incorrect":
            msg += f" (Truth: {ground_truth})"
            
    print(msg)
    return result

def main():
    parser = argparse.ArgumentParser(description="Test ONNX model output by train.py")
    parser.add_argument("--model", default="datasets/ocr/ocr_model.onnx", help="Path to ONNX model")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image to test")
    group.add_argument("--dataset_dir", help="Path to a dataset directory containing an 'images' subfolder")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    if args.image:
        run_inference(args.model, args.image)
    elif args.dataset_dir:
        images_dir = os.path.join(args.dataset_dir, "images")
        labels_file = os.path.join(args.dataset_dir, "labels.jsonl")
        
        if not os.path.exists(images_dir):
            print(f"Error: Images directory not found at {images_dir}")
            return
            
        # Load labels
        label_map = {}
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    if record.get('isCorrect', False):
                        # Replicate cleaning logic from train.py
                        label_raw = record.get('label', '')
                        label_text = "".join([c for c in label_raw if c in CHARACTERS and not c.isspace()])
                        if 2 <= len(label_text) <= 8:
                            label_map[record['filename']] = label_text
        else:
            print(f"Warning: Labels file not found at {labels_file}, skipping accuracy calculation.")

        print(f"Testing images in {images_dir}...")
        
        correct = 0
        incorrect = 0
        total = 0
        
        for filename in sorted(os.listdir(images_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                prediction = run_inference(
                    args.model, 
                    os.path.join(images_dir, filename), 
                    ground_truth=label_map.get(filename)
                )
                
                if filename in label_map:
                    total += 1
                    if prediction == label_map[filename]:
                        correct += 1
                    else:
                        incorrect += 1
                        
        print(f"\nEvaluation Results:")
        print(f"Total guesses: {total}")
        print(f"Number correct: {correct}")
        print(f"Number incorrect: {incorrect}")

if __name__ == "__main__":
    main()
