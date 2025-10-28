import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


DATA_DIR = "datasets"
SAVE_PATH = os.path.join("Py", "ML", "cnn_model.pth")
BATCH_SIZE = 8
NUM_CLASSES = 3  # 0=original, 1=tampered, 2=AI
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\n[INFO] Dataset loaded:")
    print(f"  Train images: {len(train_data)}")
    print(f"  Val images:   {len(val_data)}")
    print(f"  Classes:      {train_data.classes}\n")

    return train_loader, val_loader, train_data.classes



def train_model():
    train_loader, val_loader, class_names = load_data()

    # Base model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    print(f"[INFO] Training on {DEVICE.upper()} for {EPOCHS} epochs...")
    print(f"[INFO] Classes → {class_names}")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
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

        scheduler.step()

        train_acc = correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}")


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

        # --------------------------
        # Save best model
        # --------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"[INFO] New best model saved with acc={val_acc:.4f}")

    print(f"\n[DONE] Training complete. Best val acc: {best_acc:.4f}")
    print(f"[MODEL SAVED] -> {SAVE_PATH}")


if __name__ == "__main__":
    start_time = time.time()
    print("[START] CNN training initialized.")
    train_model()
    print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes")
