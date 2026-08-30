import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCuePhysicsLayer(nn.Module):
    """
    Computes both Lambertian Surface Shading Error and Shadow Intersection Error
    in a single fully-differentiable PyTorch module.
    """
    def __init__(self, shadow_error_threshold=5.0):
        super(MultiCuePhysicsLayer, self).__init__()
        self.shadow_threshold = shadow_error_threshold

    def compute_shading_residual(self, surface_normals, pixel_intensities):
        N_trans = torch.transpose(surface_normals, 1, 2)
        NN = torch.bmm(N_trans, surface_normals)
        
        eps = 1e-6 * torch.eye(3, device=surface_normals.device).unsqueeze(0)
        NN_inv = torch.pinverse(NN + eps)
        
        NI = torch.bmm(N_trans, pixel_intensities)
        L_hat = torch.bmm(NN_inv, NI)
        L_hat_norm = F.normalize(L_hat, p=2, dim=1)

        I_pred = torch.bmm(surface_normals, L_hat_norm)
        I_pred = torch.clamp(I_pred, min=0.0)

        residual_map = (pixel_intensities - I_pred) ** 2
        e_shading = torch.mean(residual_map, dim=1)

        normal_variance = torch.var(surface_normals, dim=1).sum(dim=-1, keepdim=True)
        shading_confidence = torch.sigmoid(normal_variance * 5.0)

        return e_shading, shading_confidence

    def compute_shadow_residual(self, object_points, shadow_points):
        batch_size, k_pairs, _ = object_points.shape

        if k_pairs < 2:
            e_shadow = torch.zeros((batch_size, 1), device=object_points.device)
            shadow_confidence = torch.zeros((batch_size, 1), device=object_points.device)
            return e_shadow, shadow_confidence

        ones = torch.ones((batch_size, k_pairs, 1), device=object_points.device)
        V_homo = torch.cat([object_points, ones], dim=-1)
        P_homo = torch.cat([shadow_points, ones], dim=-1)

        lines = torch.cross(V_homo, P_homo, dim=-1)
        norm = torch.norm(lines[:, :, :2], dim=-1, keepdim=True) + 1e-8
        lines_norm = lines / norm

        _, _, V_svd = torch.linalg.svd(lines_norm)
        S_estimated = V_svd[:, :, -1]

        S_expand = S_estimated.unsqueeze(-1)
        distances = torch.bmm(lines_norm, S_expand).squeeze(-1)
        e_shadow = torch.mean(distances ** 2, dim=-1, keepdim=True)

        shadow_confidence = torch.ones((batch_size, 1), device=object_points.device)
        return e_shadow, shadow_confidence

    def forward(self, surface_normals, pixel_intensities, object_points=None, shadow_points=None):
        e_shd, conf_shd = self.compute_shading_residual(surface_normals, pixel_intensities)

        if object_points is not None and shadow_points is not None:
            e_shd_line, conf_shd_line = self.compute_shadow_residual(object_points, shadow_points)
        else:
            e_shd_line = torch.zeros_like(e_shd)
            conf_shd_line = torch.zeros_like(conf_shd)

        physics_vector = torch.cat([e_shd, conf_shd, e_shd_line, conf_shd_line], dim=1)
        return physics_vector