import sys
import json
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import time


# ==========================
# Config & Utilities
# ==========================

MODEL_PATH = os.path.join("Py", "ML", "cnn_model.pth")
IMAGE_SIZE = (244, 244)
NUM_CLASSES = 3  # change if your fine-tuned model differs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def debug_log(msg):
    """Simple debugger-style print with timestamp"""
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}")


# ==========================
# CNN Classifier
# ==========================

class ResNetClassifier:
    def __init__(self, mode="pretrained"):
        self.mode = mode
        debug_log(f"Initializing model in mode: {mode}")
        self.model = self._load_model(mode).to(DEVICE)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        debug_log(f"Model initialized on {DEVICE} and transforms ready.")

    def _load_model(self, mode):
        debug_log("Loading model weights...")
        model = models.resnet50(weights=None if mode == "fine_tuned" else models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

        if mode == "fine_tuned":
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                debug_log(f"Loaded fine-tuned model from {MODEL_PATH}")
            else:
                debug_log(f"[WARN] Fine-tuned model not found at {MODEL_PATH}. Using untrained weights.")
        else:
            debug_log("Loaded pretrained ResNet50 (ImageNet).")

        return model

    def preprocess(self, img_path):
        debug_log(f"Preprocessing image: {img_path}")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            debug_log(f"[ERROR] Cannot open image: {e}")
            raise
        tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        debug_log(f"Image converted to tensor {tuple(tensor.shape)}")
        return tensor

    def predict(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        debug_log(f"Running prediction on {img_path}")
        img_tensor = self.preprocess(img_path)

        with torch.no_grad():
            start = time.time()
            output = self.model(img_tensor)
            duration = time.time() - start
            debug_log(f"Model inference time: {duration:.2f}s")

            probs = F.softmax(output, dim=1)[0].cpu()
            label = int(torch.argmax(probs))
            confidence = float(torch.max(probs))
            debug_log(f"Predicted label: {label} | Confidence: {confidence:.4f}")

        return {
            "label": label,
            "confidence": confidence,
            "probs": [float(p) for p in probs.tolist()],
            "inference_time": duration
        }


# ==========================
# Standalone Execution
# ==========================

if __name__ == "__main__":
    args = sys.argv[1:]
    image_path = args[0] if len(args) > 0 else None
    mode = args[1] if len(args) > 1 else "pretrained"

    if not image_path:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    debug_log("=== Starting CNN Prediction Script ===")
    classifier = ResNetClassifier(mode)

    try:
        result = classifier.predict(image_path)
        output_json = {
            "CNN_label": result["label"],
            "CNN_confidence": result["confidence"],
            "CNN_probabilities": result["probs"],
            "inference_time": result["inference_time"],
            "file_exists": True
        }
    except Exception as e:
        output_json = {"error": str(e), "file_exists": False}

    print(json.dumps(output_json))
    debug_log("=== Finished ===")
