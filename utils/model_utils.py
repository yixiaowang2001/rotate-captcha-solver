import os
import io
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.image_utils import CircleMask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 152


class Normalize(nn.Module):
    """
    Normalize a tensor image
    """
    def forward(self, x):
        return x / torch.norm(x, dim=1, keepdim=True).clamp(min=1e-8)


class CyclicMSELoss(nn.Module):
    """
    Loss function
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        cos_sim = torch.sum(pred * target, dim=1)
        cyclic_loss = torch.mean(1 - cos_sim)
        return 0.7 * mse_loss + 0.3 * cyclic_loss


class RegressionDataset(Dataset):
    """
    Build dataset
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.targets = []

        for file in os.listdir(root_dir):
            if file.endswith((".jpg", ".png", ".jpeg")):
                try:
                    angle_str = file.split('_')[1]
                    angle = math.radians(float(angle_str[3:].replace('_', '.')) % 360)
                    self.image_paths.append(os.path.join(root_dir, file))
                    self.targets.append([np.sin(angle), np.cos(angle)])
                except Exception as e:
                    print(f"Failed to parse: {file}, error: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, target


class MixedDataset(Dataset):
    """
    Mixed dataset for pseudo labeling
    """
    def __init__(self, labeled_dir, pseudo_dir, transform=None):
        self.labeled_dataset = RegressionDataset(labeled_dir, transform)
        self.pseudo_dataset = RegressionDataset(pseudo_dir, transform)
        self.transform = transform

    def __len__(self):
        return len(self.labeled_dataset) + len(self.pseudo_dataset)

    def __getitem__(self, idx):
        if idx < len(self.labeled_dataset):
            return self.labeled_dataset[idx]
        else:
            return self.pseudo_dataset[idx - len(self.labeled_dataset)]


class ImageFolderInfer(Dataset):
    """
    Image folder dataset for inference
    """
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.image_files = []

        if isinstance(folder_path, list):
            self.image_files = folder_path
        elif os.path.isdir(folder_path):
            self.image_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        else:
            raise ValueError("folder_path must be a directory or a list of file paths")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)


def transformation():
    """
    Transform data
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        CircleMask(IMG_SIZE),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        CircleMask(IMG_SIZE),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def build_model(dropout_rate, pretrained=True, freeze_layers=False):
    """
    Build model (based on ConvNeXt)
    """
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(768, 512),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, 2),
        Normalize()
    )

    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        for block in model.features[-3:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def load_model(model_path, dropout_rate, freeze_layers=False, device=DEVICE):
    """
    Load model
    """
    model = build_model(dropout_rate=dropout_rate, pretrained=False, freeze_layers=freeze_layers)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model.to(device)


def evaluate(model, loader):
    """
    Evaluate model
    """
    model.eval()
    total_mae = 0.0
    total_loss = 0.0
    criterion = CyclicMSELoss()

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            pred_deg = torch.rad2deg(torch.atan2(outputs[:, 0], outputs[:, 1])).cpu().numpy()
            true_deg = torch.rad2deg(torch.atan2(targets[:, 0], targets[:, 1])).cpu().numpy()

            diff = np.abs(pred_deg - true_deg)
            mae = np.mean(np.minimum(diff, 360 - diff))
            total_mae += mae * images.size(0)

    return total_mae / len(loader.dataset), total_loss / len(loader)


def train_and_evaluate(
        model,
        train_loader,
        val_loader,
        model_name,
        epochs,
        feature_lr=None,
        cls_lr=None,
        weight_decay=1e-4,
        device=DEVICE,
        use_plateau_scheduler=False
):
    """
    Train and evaluate model with flexible optimizer and scheduler choices.
    """
    model = model.to(device)
    criterion = CyclicMSELoss()

    if feature_lr and cls_lr:
        optimizer = optim.AdamW([
            {"params": model.features[-3:].parameters(), "lr": feature_lr},
            {"params": model.classifier.parameters(), "lr": cls_lr}
        ], weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cls_lr or 1e-3,
            weight_decay=weight_decay
        )

    if use_plateau_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cls_lr or 1e-3,
            steps_per_epoch=len(train_loader),
            epochs=epochs
        )

    best_mae = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Training"):
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            if not use_plateau_scheduler:
                scheduler.step()

            train_loss += loss.item()

        val_mae, val_loss = evaluate(model, val_loader)
        train_loss /= len(train_loader)

        if use_plateau_scheduler:
            scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.2f}°")

        if val_mae < best_mae * 1.1:
            best_mae = val_mae
            torch.save(model.state_dict(), f"models/{model_name}.pth")
            print(f"New model saved: {model_name}.pth (MAE: {best_mae:.2f}°)")

    print(f"\n{model_name} finished training. Best val MAE: {best_mae:.2f}°")
    return best_mae


def predict_angle(image_input, model, transform, input_type='path'):
    """
    Predict angle based on input (image path or image bytes)
    """
    try:
        if input_type == "path":
            image = Image.open(image_input).convert("RGB")
        elif input_type == "bytes":
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            raise ValueError("input_type must be 'path' or 'bytes'")

        tensor = transform(image).unsqueeze(0).to(DEVICE)
        model.eval()
        with torch.no_grad():
            output = model(tensor)
        sin_pred, cos_pred = output[0].cpu().numpy()
        return math.degrees(math.atan2(sin_pred, cos_pred)) % 360

    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return None


def predict_folder(model, folder_path, transform, batch_size=64):
    """
    Predict angles based on input folder
    """
    dataset = ImageFolderInfer(folder_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = {}
    model.eval()
    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(DEVICE)
            outputs = model(images)

            sin_pred = outputs[:, 0].cpu().numpy()
            cos_pred = outputs[:, 1].cpu().numpy()
            angles_deg = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360

            for filename, angle in zip(filenames, angles_deg):
                results[filename] = angle

    return results
