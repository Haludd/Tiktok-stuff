import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class AIDetectionPhysicsDataset(Dataset):
    """
    PyTorch Dataset for Physics-Enhanced AI Image Detection.
    Loads RGB images, extracts 3D surface normals and shadow points dynamically,
    and returns paired tensors ready for the training pipeline.
    """
    def __init__(self, root_dir, split="train", num_sample_points=200):
        """
        Args:
            root_dir (str): Root directory containing 'real' and 'fake' subfolders.
            split (str): 'train', 'val', or 'test'.
            num_sample_points (int): Number of surface normal points to sample per image.
        """
        super(AIDetectionPhysicsDataset, self).__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.num_points = num_sample_points

        # Image Transformation for standard ResNet backbone (224x224 RGB)
        self.rgb_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load file paths: Real (0) vs Fake (1)
        self.image_paths = []
        self.labels = []

        real_dir = os.path.join(self.root_dir, "real")
        fake_dir = os.path.join(self.root_dir, "fake")

        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(real_dir, img_name))
                    self.labels.append(0) # Label 0 = Real

        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(fake_dir, img_name))
                    self.labels.append(1) # Label 1 = Fake / AI-Generated

    def _extract_surface_geometry(self, img_bgr):
        """
        Extracts surface normals and intensities from image gradients.
        (Note: In production, replace the gradient approximation with a pre-trained 
        depth model like MiDaS or Marigold for full 3D precision).
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0

        # Compute Sobel Gradients (dX, dY) as 3D surface normal approximations
        gx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)
        gz = np.ones_like(gray_norm)

        # Construct 3D Normal Map: N = (gx, gy, gz) / norm
        normals_map = np.stack([-gx, -gy, gz], axis=-1)
        norm = np.linalg.norm(normals_map, axis=-1, keepdims=True) + 1e-8
        normals_map = normals_map / norm

        # Sample K uniformly distributed points across the image plane
        h, w = gray.shape
        idx_y = np.random.randint(0, h, size=self.num_points)
        idx_x = np.random.randint(0, w, size=self.num_points)

        sampled_normals = normals_map[idx_y, idx_x, :] # Shape: (num_points, 3)
        sampled_intensities = gray_norm[idx_y, idx_x, None] # Shape: (num_points, 1)

        return torch.tensor(sampled_normals, dtype=torch.float32), \
               torch.tensor(sampled_intensities, dtype=torch.float32)

    def _extract_shadow_pairs(self, img_bgr):
        """
        Extracts candidate object base (V) and shadow tip (P) point pairs.
        Uses thresholding and contour direction vectors as heuristic extraction.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Dark regions thresholding for potential cast shadows
        _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obj_pts, shd_pts = [], []

        for cnt in contours:
            if cv2.contourArea(cnt) > 200: # Filter small noise contours
                # Fit oriented bounding box to shadow contour
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                
                # Take opposite corners of the bounding box as vector line (v -> p)
                v_point = box[0] # Object Base approximation
                p_point = box[2] # Shadow Tip approximation

                obj_pts.append(v_point)
                shd_pts.append(p_point)

                if len(obj_pts) >= 4: # Limit to max 4 prominent shadow pairs per frame
                    break

        # Fallback if no valid shadows are found
        if len(obj_pts) < 2:
            obj_tensor = torch.zeros((0, 2), dtype=torch.float32)
            shd_tensor = torch.zeros((0, 2), dtype=torch.float32)
        else:
            obj_tensor = torch.tensor(np.array(obj_pts), dtype=torch.float32)
            shd_tensor = torch.tensor(np.array(shd_pts), dtype=torch.float32)

        return obj_tensor, shd_tensor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 1. Load Image
        img_pil = Image.open(img_path).convert("RGB")
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            # Handle corrupted image files
            img_bgr = np.array(img_pil)[:, :, ::-1]

        # 2. Extract Standard RGB Tensor for ResNet Backbone
        rgb_tensor = self.rgb_transform(img_pil)

        # 3. Extract Surface Normals & Pixel Intensities
        normals_tensor, intensities_tensor = self._extract_surface_geometry(img_bgr)

        # 4. Extract Shadow Point Pairs
        obj_pts_tensor, shd_pts_tensor = self._extract_shadow_pairs(img_bgr)

        return {
            "rgb": rgb_tensor,
            "normals": normals_tensor,
            "intensities": intensities_tensor,
            "obj_pts": obj_pts_tensor,
            "shd_pts": shd_pts_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }