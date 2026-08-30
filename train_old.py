import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from dataset import AIDetectionPhysicsDataset

# Import your custom physics module from step 1
from physics_layer import MultiCuePhysicsLayer

# ---------------------------------------------------------
# 1. Hardware & Global Configuration
# ---------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "physics_enhanced_detector.pth"


# ---------------------------------------------------------
# 2. Complete Model Architecture (Backbone + Physics)
# ---------------------------------------------------------
class PhysicsEnhancedDetector(nn.Module):
    def __init__(self, use_physics=True):
        super(PhysicsEnhancedDetector, self).__init__()
        self.use_physics = use_physics

        # A. Pre-trained Visual Backbone (ResNet-50)
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Strip standard 1000-class head

        # Freeze ResNet backbone during initial training phase
        for param in self.backbone.parameters():
            param.requires_grad = False

        # B. Custom Differentiable Physics Layer
        self.physics_layer = MultiCuePhysicsLayer()

        # C. Fusion Classifier Head
        input_dim = num_ftrs + (4 if self.use_physics else 0)

        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Logits output
        )

    def forward(self, rgb_imgs, normals, intensities, obj_pts=None, shd_pts=None):
        v_rgb = self.backbone(rgb_imgs)  # Shape: (Batch, 2048)

        if self.use_physics:
            v_phys = self.physics_layer(normals, intensities, obj_pts, shd_pts) # Shape: (Batch, 4)
            fused_features = torch.cat([v_rgb, v_phys], dim=1)
        else:
            fused_features = v_rgb

        logits = self.classifier_head(fused_features)
        return logits


# ---------------------------------------------------------
# 3. Training Engine Loop with Progress Bar
# ---------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epochs):
    model.train()
    running_loss, correct_preds, total_samples = 0.0, 0, 0

    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch}/{total_epochs}] Physics Training", leave=True)

    for batch in progress_bar:
        rgb = batch["rgb"].to(DEVICE)
        normals = batch["normals"].to(DEVICE)
        intensities = batch["intensities"].to(DEVICE)
        obj_pts = batch["obj_pts"].to(DEVICE) if "obj_pts" in batch else None
        shd_pts = batch["shd_pts"].to(DEVICE) if "shd_pts" in batch else None
        labels = batch["label"].to(DEVICE).unsqueeze(1).float()

        optimizer.zero_grad()

        logits = model(rgb, normals, intensities, obj_pts, shd_pts)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits) > 0.5).float()
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)
        running_loss += loss.item() * labels.size(0)

        current_loss = running_loss / total_samples
        current_acc = correct_preds / total_samples
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc*100:.2f}%")

    return running_loss / total_samples, correct_preds / total_samples


# ---------------------------------------------------------
# 4. Validation Engine Loop with Progress Bar
# ---------------------------------------------------------
def validate(model, dataloader, criterion, epoch, total_epochs):
    model.eval()
    running_loss, correct_preds, total_samples = 0.0, 0, 0

    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch}/{total_epochs}] Physics Validation", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            rgb = batch["rgb"].to(DEVICE)
            normals = batch["normals"].to(DEVICE)
            intensities = batch["intensities"].to(DEVICE)
            obj_pts = batch["obj_pts"].to(DEVICE) if "obj_pts" in batch else None
            shd_pts = batch["shd_pts"].to(DEVICE) if "shd_pts" in batch else None
            labels = batch["label"].to(DEVICE).unsqueeze(1).float()

            logits = model(rgb, normals, intensities, obj_pts, shd_pts)
            loss = criterion(logits, labels)

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)
            running_loss += loss.item() * labels.size(0)

            current_loss = running_loss / total_samples
            current_acc = correct_preds / total_samples
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc*100:.2f}%")

    return running_loss / total_samples, correct_preds / total_samples

def custom_collate(batch):
        rgb = torch.stack([item['rgb'] for item in batch])
        normals = torch.stack([item['normals'] for item in batch])
        intensities = torch.stack([item['intensities'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])

        obj_pts_list = [item['obj_pts'] for item in batch]
        shd_pts_list = [item['shd_pts'] for item in batch]
        
        max_k = max([pts.shape[0] for pts in obj_pts_list])
        if max_k == 0:
            max_k = 1
            
        padded_obj_pts = []
        padded_shd_pts = []
        for o_pts, s_pts in zip(obj_pts_list, shd_pts_list):
            k = o_pts.shape[0]
            if k < max_k:
                pad_o = torch.zeros((max_k - k, 2), dtype=torch.float32)
                pad_s = torch.zeros((max_k - k, 2), dtype=torch.float32)
                o_pts = torch.cat([o_pts, pad_o], dim=0)
                s_pts = torch.cat([s_pts, pad_s], dim=0)
            elif k == 0:
                o_pts = torch.zeros((max_k, 2), dtype=torch.float32)
                s_pts = torch.zeros((max_k, 2), dtype=torch.float32)
            padded_obj_pts.append(o_pts)
            padded_shd_pts.append(s_pts)

        obj_pts = torch.stack(padded_obj_pts)
        shd_pts = torch.stack(padded_shd_pts)

        return {
            "rgb": rgb,
            "normals": normals,
            "intensities": intensities,
            "obj_pts": obj_pts,
            "shd_pts": shd_pts,
            "label": labels
        }

# ---------------------------------------------------------
# 5. Main Execution & Training Orchestrator
# ---------------------------------------------------------
def main():
    DATASET_PATH = "./dataset_root"

    train_dataset = AIDetectionPhysicsDataset(root_dir=DATASET_PATH, split="train")
    val_dataset = AIDetectionPhysicsDataset(root_dir=DATASET_PATH, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, num_workers=7, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate, num_workers=7, pin_memory=True)

    print(f"Dataset Loaded Successfully! Total Training Samples: {len(train_dataset)}")

    model = PhysicsEnhancedDetector(use_physics=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    print("\n--- Starting Physics-Enhanced Training ---")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch + 1, EPOCHS)
        val_loss, val_acc = validate(model, val_loader, criterion, epoch + 1, EPOCHS)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Summary | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved successfully to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()