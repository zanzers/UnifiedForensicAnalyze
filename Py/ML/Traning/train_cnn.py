import os
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 400_000_000
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time


# SETTING !
DATA_DIR = "Cnn_datasets" 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(BASE_DIR)
SAVE_PATH = os.path.join(ML_DIR, "cnn_model.pth")


BATCH_SIZE = 8
NUM_CLASSES = 3
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# start loading data
def load_data():
    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size

    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[INFO] Loaded {train_size} training and {val_size} validation images.")
    print(f"[INFO] Classes found: {full_dataset.classes}")
    return train_loader, val_loader, full_dataset.classes


def train_model():

    if os.path.exists(SAVE_PATH):
        print(f"[INFO] Existing model found at {SAVE_PATH}. Deleting old model...")
        os.remove(SAVE_PATH)



    train_loader, val_loader, class_names = load_data()

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[INFO] Training on {DEVICE} for {EPOCHS} epochs...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={train_acc:.4f}")

     
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.4f}")


        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"[INFO] New best model saved with acc={val_acc:.4f}")

    print(f"[DONE] Training complete. Best val acc: {best_acc:.4f}")
    print(f"[MODEL SAVED] -> {SAVE_PATH}")




if __name__ == "__main__":
    start_time = time.time()
    train_model()
    print(f"Total training time: {(time.time()-start_time)/60:.2f} minutes")



