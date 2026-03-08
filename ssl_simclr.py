from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    """SimCLR model: backbone encoder + projection head."""

    def __init__(self, backbone: nn.Module, feat_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #backbone trả về features, không phải logits
        feats = self.backbone(x, return_features=True)
        z = self.projector(feats)
        z = F.normalize(z, dim=1)
        return z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    
    assert z1.shape == z2.shape
    bsz = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    sim = (z @ z.t()).float() / temperature  # [2B, 2B]

    diag = torch.eye(2 * bsz, device=sim.device, dtype=torch.bool)
    sim.masked_fill_(diag, float("-inf"))

    pos = torch.cat([torch.diag(sim, bsz), torch.diag(sim, -bsz)], dim=0)  # [2B]

    denom = torch.logsumexp(sim, dim=1)  # [2B]

    loss = (-pos + denom).mean()
    return loss
