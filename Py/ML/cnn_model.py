import sys
import json
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import time


def debug_log(msg):
    """Simple debugger-style print with timestamp"""
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}")


class ResNetClassifier:
    def __init__(self, mode="pretrained"):
        self.mode = mode
        debug_log(f"Initializing model in mode: {mode}")
        self.model = self._load_model(mode)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        debug_log("Model initialized and transforms ready.")

    def _load_model(self, mode):
        debug_log("Loading model...")
        if mode == "fine_tuned":
            model_path = os.path.join("Py", "ML", "cnn_model.pth")
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 3)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
                debug_log(f"Loaded fine-tuned model from {model_path}")
            else:
                debug_log(f"[WARN] Fine-tuned model not found at {model_path}. Using untrained weights.")
        else:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            debug_log("Loaded pretrained ResNet50 (ImageNet).")

        debug_log("Model loading complete.")
        return model

    def preprocess(self, img_path):
        debug_log(f"Preprocessing image: {img_path}")
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        debug_log(f"Image preprocessed into tensor: {tuple(tensor.shape)}")
        return tensor

    def predict(self, img_path):
        debug_log(f"Running prediction for: {img_path}")
        img_tensor = self.preprocess(img_path)
        with torch.no_grad():
            start = time.time()
            output = self.model(img_tensor)
            end = time.time()
            debug_log(f"Model forward pass took: {end - start:.2f}s")

            probs = F.softmax(output, dim=1)[0]
            label = int(torch.argmax(probs).item())
            debug_log(f"Predicted label: {label}")
        return label, probs.tolist()


if __name__ == "__main__":
    args = sys.argv[1:]
    image_path = args[0] if len(args) > 0 else None
    mode = "pretrained"

    debug_log("Starting main execution.")
    model = ResNetClassifier(mode)

    if not os.path.exists(image_path):
        print(json.dumps({"error": "Image not found"}))
        sys.exit(1)

    label, probs = model.predict(image_path)

    result = {
        "CNN_label": label,
        "CNN_prob_0": float(probs[0]),
        "CNN_prob_1": float(probs[1]),
        "CNN_prob_2": float(probs[2]),
        "file_exists": os.path.exists(image_path)
    }

    debug_log("Execution complete. Returning result.")
    print(json.dumps(result))
