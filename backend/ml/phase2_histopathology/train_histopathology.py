"""
Histopathology Model Training — EfficientNet-B4 on OSCC H&E slides
Dataset structure expected at data/raw/histopathology/:
  train/
    Normal/
    OSCC/
  val/
    Normal/
    OSCC/
  test/
    Normal/
    OSCC/
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "histopathology")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts", "models")

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 3e-4
IMAGE_SIZE = 224


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    print("=== Histopathology EfficientNet-B4 Training ===")

    # Validate dataset structure
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            print(f"Error: Missing dataset folder: {d}")
            print(f"Please unzip the Kaggle dataset into {DATA_DIR}/")
            return

    # --- Data Transforms ---
    # H&E stained slides: we apply aggressive augmentation to handle stain variability
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        # ImageNet normalization is the standard for pre-trained models
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Datasets ---
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)

    print(f"Classes: {train_dataset.classes}  →  {train_dataset.class_to_idx}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = get_device()
    print(f"Device: {device}")

    # --- Model: EfficientNet-B4 (best accuracy on this dataset) ---
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

    # Freeze early feature layers, only fine-tune the top layers
    for param in model.features[:6].parameters():
        param.requires_grad = False

    # Replace the final classifier for binary classification
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, 1)
    )
    model = model.to(device)

    # --- Loss, Optimizer, Scheduler ---
    # Determine class weights to handle imbalanced datasets (more OSCC than Normal)
    oscc_count = len(os.listdir(os.path.join(train_dir, "OSCC"))) if os.path.exists(os.path.join(train_dir, "OSCC")) else 1
    normal_count = len(os.listdir(os.path.join(train_dir, "Normal"))) if os.path.exists(os.path.join(train_dir, "Normal")) else 1
    pos_weight = torch.tensor([normal_count / oscc_count], dtype=torch.float32).to(device)
    print(f"Class balance → Normal: {normal_count} | OSCC: {oscc_count} | pos_weight: {pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- Training Loop ---
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_save_path = os.path.join(ARTIFACTS_DIR, "histopathology_efficientnet_b4.pt")
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            # ImageFolder: OSCC=0, Normal=1 if alphabetical. We need OSCC=1 (positive).
            # class_to_idx will tell us the actual mapping. Let's handle it:
            oscc_idx = train_dataset.class_to_idx.get("OSCC", 0)
            labels_binary = (labels == oscc_idx).float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_binary)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels_binary).sum().item()
            total += inputs.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels_binary = (labels == oscc_idx).float().unsqueeze(1).to(device)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels_binary).sum().item()
                val_total += inputs.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f" --> Saved best model (Val Acc: {val_acc:.4f}) to {model_save_path}")

    # --- Final Test Evaluation ---
    print("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels_binary = (labels == oscc_idx).float().unsqueeze(1).to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            test_correct += (preds == labels_binary).sum().item()
            test_total += inputs.size(0)

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"\nTraining complete! Best model saved at: {model_save_path}")


if __name__ == "__main__":
    main()
