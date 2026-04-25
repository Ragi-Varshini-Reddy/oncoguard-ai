import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Ensure robust path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "intraoral")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts", "models")

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4


def rgb_loader(path):
    with open(path, "rb") as handle:
        with Image.open(handle) as image:
            rgb_image = image.convert("RGB")
            rgb_image.load()
            return rgb_image


class ImageSamplesDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = rgb_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, target


def stratified_split(samples, train_fraction=0.8, seed=42):
    rng = random.Random(seed)
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample[1], []).append(sample)

    train_samples = []
    val_samples = []
    for label_samples in grouped.values():
        rng.shuffle(label_samples)
        split_at = max(1, int(len(label_samples) * train_fraction))
        if split_at >= len(label_samples) and len(label_samples) > 1:
            split_at = len(label_samples) - 1
        train_samples.extend(label_samples[:split_at])
        val_samples.extend(label_samples[split_at:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    print("Checking intraoral dataset...")
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory not found at {DATA_DIR}")
        return

    # Advanced Data Augmentation for clinical images
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset
    try:
        full_dataset = datasets.ImageFolder(root=DATA_DIR, loader=rgb_loader)
    except FileNotFoundError:
        print(f"Error: No images found in {DATA_DIR}. Make sure you placed them in CANCER and NON CANCER subfolders.")
        return

    if len(full_dataset) == 0:
        print(f"Error: Dataset found but contains 0 images.")
        return

    class_names = full_dataset.classes
    print(f"Found classes: {class_names}")
    if "CANCER" not in full_dataset.class_to_idx or "NON CANCER" not in full_dataset.class_to_idx:
        print("Error: Expected class folders named CANCER and NON CANCER.")
        return

    train_samples, val_samples = stratified_split(full_dataset.samples)
    train_size = len(train_samples)
    val_size = len(val_samples)
    train_dataset = ImageSamplesDataset(train_samples, train_transform)
    val_dataset = ImageSamplesDataset(val_samples, val_transform)
    print(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = get_device()
    print(f"Using device: {device}")

    # Load Pre-trained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Modify final layer for Binary Classification
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    cancer_idx = full_dataset.class_to_idx["CANCER"]
    positive_count = sum(1 for _, label in train_samples if label == cancer_idx)
    negative_count = max(1, train_size - positive_count)
    pos_weight = torch.tensor([negative_count / max(1, positive_count)], device=device)

    # Use BCEWithLogitsLoss since output is a single logit (unscaled probability)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_loss = float('inf')
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_save_path = os.path.join(ARTIFACTS_DIR, "intraoral_efficientnet.pt")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = (labels == cancer_idx).float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / train_size
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = (labels == cancer_idx).float().unsqueeze(1).to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = val_loss / val_size
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")

        # Save Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f" --> Saved improved model to {model_save_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()
