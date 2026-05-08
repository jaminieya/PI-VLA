import torch
from torch import nn
from torchvision import models
from typing import Optional


def build_resnet18():
    if hasattr(models, "ResNet18_Weights"):
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return models.resnet18(pretrained=True)


class TextPromptEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, token_ids):
        emb = self.embed(token_ids)
        mask = (token_ids != 0).unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (emb * mask).sum(dim=1) / denom


class RegressionStudent(nn.Module):
    """
    Deterministic student model for direct z_goal regression.

    Given image + text, predicts a single z_goal vector via MSE.
    No distribution learning — one forward pass, one prediction.

    Training:  forward(x, text_tokens, z_goal) → (mse_loss, dummy_zero)
    Inference: forward(x, text_tokens)          → (z_pred,   dummy_zero)
    """

    _TEXT_DIM = 64

    def __init__(
        self,
        output_dim: int,
        vocab_size: int = 1,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.output_dim = output_dim

        # ── Backbone ─────────────────────────────────────────────────────────
        backbone = build_resnet18()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        td = self._TEXT_DIM
        self.text_encoder = TextPromptEncoder(vocab_size=vocab_size, embed_dim=td)

        # ── Regression head ───────────────────────────────────────────────────
        # 512 (ResNet) + 64 (text) = 576 → 512 → 256 → output_dim
        cond_dim = 512 + td
        self.regressor = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p / 2),
            nn.Linear(256, output_dim),
        )

        self.criterion = nn.MSELoss()

    # ── Image features ────────────────────────────────────────────────────────
    def _img_features(self, x: torch.Tensor) -> torch.Tensor:
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if self.training and backbone_trainable:
            return self.backbone(x)
        with torch.no_grad():
            return self.backbone(x)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        z_goal: Optional[torch.Tensor] = None,
    ):
        """
        Training (z_goal provided):  returns (mse_loss, dummy_zero)
        Inference (z_goal=None):     returns (z_pred,   dummy_zero)
        """
        legacy_no_text_call = text_tokens is None
        if legacy_no_text_call:
            # Backward compatibility for no-text training scripts.
            text_tokens = torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device)

        img_feat  = self._img_features(x)                   # (B, 512)
        text_feat = self.text_encoder(text_tokens)           # (B, 64)
        cond      = torch.cat([img_feat, text_feat], dim=1)  # (B, 576)
        z_pred    = self.regressor(cond)                     # (B, D)

        if z_goal is not None:
            loss = self.criterion(z_pred, z_goal)
            return loss, torch.zeros((), device=x.device, dtype=x.dtype)

        if legacy_no_text_call:
            return z_pred

        return z_pred, torch.zeros((), device=x.device, dtype=x.dtype)

    # ── API compatibility shims ───────────────────────────────────────────────
    def predict_best(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic prediction — identical to forward at inference."""
        z_pred, _ = self.forward(x, text_tokens)
        return z_pred

    def get_multiple_latent_predictions(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
        num_samples: int = 10,
        z_goal=None,
    ) -> torch.Tensor:
        """
        Returns (num_samples, B, D) with the same prediction repeated.
        Keeps the interface compatible with MDNStudent-based planning loops.
        """
        self.eval()
        with torch.no_grad():
            z_pred, _ = self.forward(x, text_tokens)        # (B, D)
        return z_pred.unsqueeze(0).expand(num_samples, -1, -1)

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total     = trainable + frozen
        print(
            f"Parameters | trainable: {trainable:,}  "
            f"frozen: {frozen:,}  total: {total:,}",
            flush=True,
        )
        return trainable, frozen, total