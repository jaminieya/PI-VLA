import math
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


class MDNStudent(nn.Module):
    """
    Mixture Density Network for multimodal z_goal prediction.

    Given image + text, predicts a K-component Gaussian mixture over z_goal.
    Each component has its own mean (mu) and isotropic variance (sigma).

    Training: minimizes NLL of z_goal under the mixture.
    Inference: samples from the mixture (or returns the mean of the
               highest-weight component for deterministic planning).

    At inference time, call forward(x, text_tokens) → (z_pred, dummy_loss).
    For best-of-K planning, call get_multiple_latent_predictions(...).
    """

    _TEXT_DIM = 64

    def __init__(
        self,
        output_dim: int,
        vocab_size: int,
        n_components: int = 8,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_components = n_components

        # ── Backbone ────────────────────────────────────────────────────────
        backbone = build_resnet18()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Unfreeze layer4 for task-specific adaptation
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        td = self._TEXT_DIM
        self.text_encoder = TextPromptEncoder(vocab_size=vocab_size, embed_dim=td)

        # ── Shared feature network ───────────────────────────────────────────
        # 512 (ResNet) + 64 (text) = 576 → 512 → 256
        cond_dim = 512 + td
        self.feature_net = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p / 2),
        )

        # ── Mixture heads ────────────────────────────────────────────────────
        K, D = n_components, output_dim
        self.pi_head    = nn.Linear(256, K)               # mixture logits
        self.mu_head    = nn.Linear(256, K * D)           # component means
        self.sigma_head = nn.Linear(256, K)               # per-component log-sigma (isotropic)

    # ── Image features ───────────────────────────────────────────────────────
    def _img_features(self, x: torch.Tensor) -> torch.Tensor:
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if self.training and backbone_trainable:
            return self.backbone(x)
        with torch.no_grad():
            return self.backbone(x)

    # ── Mixture parameters ───────────────────────────────────────────────────
    def _mixture_params(self, x: torch.Tensor, text_tokens: torch.Tensor):
        """
        Returns (pi, mu, sigma):
          pi:    (B, K)       mixture weights (softmax, sums to 1)
          mu:    (B, K, D)    component means
          sigma: (B, K, 1)    per-component std (isotropic, > 0)
        """
        img_feat  = self._img_features(x)                        # (B, 512)
        text_feat = self.text_encoder(text_tokens)                # (B, 64)
        cond      = torch.cat([img_feat, text_feat], dim=1)       # (B, 576)
        h         = self.feature_net(cond)                        # (B, 256)

        B, K, D = x.size(0), self.n_components, self.output_dim

        pi    = torch.softmax(self.pi_head(h), dim=-1)            # (B, K)
        mu    = self.mu_head(h).view(B, K, D)                     # (B, K, D)
        # sigma must be positive; clamp log-sigma for numerical stability
        sigma = torch.exp(
            self.sigma_head(h).clamp(-6.0, 3.0)
        ).unsqueeze(-1)                                           # (B, K, 1)

        return pi, mu, sigma

    # ── NLL loss ─────────────────────────────────────────────────────────────
    @staticmethod
    def _nll_loss(
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative log-likelihood of target under the Gaussian mixture.

        pi:     (B, K)
        mu:     (B, K, D)
        sigma:  (B, K, 1)   isotropic std per component
        target: (B, D)
        """
        D = target.size(-1)
        target_exp = target.unsqueeze(1).expand_as(mu)            # (B, K, D)

        # Log prob of each component: sum over D dims
        # log N(x | mu_k, sigma_k^2 I)
        log_prob = (
            -0.5 * D * math.log(2 * math.pi)
            - D * sigma.squeeze(-1).log()                         # (B, K)
            - 0.5 * ((target_exp - mu) / sigma).pow(2).sum(-1)   # (B, K)
        )

        log_pi  = torch.log(pi + 1e-8)                            # (B, K)
        log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)      # (B,)
        return -log_mix.mean()

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
        z_goal: Optional[torch.Tensor] = None,
    ):
        """
        Training (z_goal provided):
            returns (nll_loss, dummy_zero)
            — plug into the existing training loop as (z_pred, kl_loss)
              where z_pred is ignored (loss comes from nll_loss).

        Inference (z_goal=None):
            returns (z_pred, dummy_zero)
            z_pred is sampled from the mixture.
        """
        pi, mu, sigma = self._mixture_params(x, text_tokens)

        if z_goal is not None:
            loss = self._nll_loss(pi, mu, sigma, z_goal)
            # Return loss as first element so run_batches can call criterion(z_pred, y)
            # We override this in the training script — see full_train_multi_mdn.py
            return loss, torch.zeros((), device=x.device, dtype=x.dtype)

        # Inference: sample a component index, then sample from that Gaussian
        # For deterministic planning use the highest-weight component mean instead.
        k_idx = torch.distributions.Categorical(pi).sample()      # (B,)
        B, K, D = mu.shape
        mu_k    = mu[torch.arange(B, device=mu.device), k_idx]    # (B, D)
        sigma_k = sigma[torch.arange(B, device=mu.device), k_idx] # (B, 1)
        z_pred  = mu_k + sigma_k * torch.randn_like(mu_k)
        return z_pred, torch.zeros((), device=x.device, dtype=x.dtype)

    # ── deterministic inference (best component mean) ─────────────────────────
    def predict_best(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean of the highest-weight component (deterministic)."""
        pi, mu, _ = self._mixture_params(x, text_tokens)
        k_best = pi.argmax(dim=-1)                                 # (B,)
        B = mu.size(0)
        return mu[torch.arange(B, device=mu.device), k_best]      # (B, D)

    # ── best-of-K planning samples ────────────────────────────────────────────
    def get_multiple_latent_predictions(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
        num_samples: int = 10,
        z_goal=None,                  # unused at inference, kept for API compat
    ) -> torch.Tensor:
        """
        Draw num_samples independent z_goal candidates from the mixture.
        Returns (num_samples, B, D) — use each row as a planning goal and
        pick the one that leads to the best planner outcome.
        """
        self.eval()
        with torch.no_grad():
            pi, mu, sigma = self._mixture_params(x, text_tokens)
        B, K, D = mu.shape
        preds = []
        for _ in range(num_samples):
            k_idx   = torch.distributions.Categorical(pi).sample()
            mu_k    = mu[torch.arange(B, device=mu.device), k_idx]
            sigma_k = sigma[torch.arange(B, device=mu.device), k_idx]
            z_pred  = mu_k + sigma_k * torch.randn_like(mu_k)
            preds.append(z_pred)
        return torch.stack(preds, dim=0)                           # (S, B, D)

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