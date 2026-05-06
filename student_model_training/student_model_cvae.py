import torch
from torch import nn
from torchvision import models


def build_resnet18():
    """Build ResNet18 with pretrained weights across torchvision versions."""
    if hasattr(models, "ResNet18_Weights"):
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return models.resnet18(pretrained=True)


class TextPromptEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, token_ids):
        emb = self.embed(token_ids)          # (B, T, D)
        mask = (token_ids != 0).unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (emb * mask).sum(dim=1) / denom


class CVAEStudent(nn.Module):
    """CVAE with learned Gaussian prior p(z|x, text)."""

    # Image (512) + pooled text embedding — keep in sync with TextPromptEncoder.embed_dim
    _TEXT_DIM = 64

    def __init__(self, output_dim, vocab_size, latent_dim=128, dropout_p: float = 0.2):
        super().__init__()
        self.latent_dim = int(latent_dim)
        backbone = build_resnet18()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Task-specific visual adaptation (ImageNet trunk stays frozen except layer4)
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        td = self._TEXT_DIM
        self.text_encoder = TextPromptEncoder(vocab_size=vocab_size, embed_dim=td)

        # Encoder: image + text + z_goal → posterior q(z|x, z_goal, text)
        enc_in = 512 + td + output_dim
        enc_hidden = 256
        self.encoder_mu = nn.Sequential(
            nn.Linear(enc_in, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, self.latent_dim),
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(enc_in, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, self.latent_dim),
        )

        # Decoder: z only → z_goal (no cond) so z must carry the reconstruction signal (anti–posterior-collapse)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, output_dim),
        )

        # Prior p(z|x, text) — small MLP; inference samples z via reparameterize(prior_mu, prior_logvar)
        cond_dim = 512 + td
        prior_hidden = 256
        prior_do = 0.1
        self.prior_mu = nn.Sequential(
            nn.Linear(cond_dim, prior_hidden),
            nn.ReLU(),
            nn.Dropout(p=prior_do),
            nn.Linear(prior_hidden, self.latent_dim),
        )
        self.prior_logvar = nn.Sequential(
            nn.Linear(cond_dim, prior_hidden),
            nn.ReLU(),
            nn.Dropout(p=prior_do),
            nn.Linear(prior_hidden, self.latent_dim),
        )

    def encode(self, img_feat, text_feat, z_goal):
        h = torch.cat([img_feat, text_feat, z_goal], dim=1)
        return self.encoder_mu(h), self.encoder_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_gaussian_diagonal(mu, logvar, prior_mu, prior_logvar) -> torch.Tensor:
        """KL(q||p) for diagonal Gaussians q=N(mu,·), p=N(prior_mu,·)."""
        # Numerical stability on variance ratio term
        prior_var = prior_logvar.exp()
        inner = (logvar.exp() + (mu - prior_mu).pow(2)) / (prior_var + 1e-8)
        per_elem = prior_logvar - logvar + inner - 1.0
        return 0.5 * per_elem.mean()

    def _img_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone forward; enables grads only when training unfrozen layers."""
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if self.training and backbone_trainable:
            return self.backbone(x)
        with torch.no_grad():
            return self.backbone(x)

    def _forward_with_kl(self, x, text_tokens, z_goal=None, return_aux: bool = False):
        img_feat = self._img_features(x)
        text_feat = self.text_encoder(text_tokens)
        cond = torch.cat([img_feat, text_feat], dim=1)

        prior_mu = self.prior_mu(cond)
        prior_logvar = self.prior_logvar(cond)

        posterior_mu = None
        if z_goal is not None:
            mu, logvar = self.encode(img_feat, text_feat, z_goal)
            z = self.reparameterize(mu, logvar)
            kl_loss = self._kl_gaussian_diagonal(mu, logvar, prior_mu, prior_logvar)
            posterior_mu = mu
        else:
            # Inference: sample from prior p(z|x, text)
            z = self.reparameterize(prior_mu, prior_logvar)
            kl_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        z_pred = self.decoder(z)
        if return_aux:
            return z_pred, kl_loss, {"prior_mu": prior_mu, "posterior_mu": posterior_mu}
        return z_pred, kl_loss

    def forward(self, x, text_tokens, z_goal=None, return_aux: bool = False):
        out = self._forward_with_kl(x, text_tokens, z_goal=z_goal, return_aux=return_aux)
        return out

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        print(
            f"Parameters | trainable: {trainable:,}  "
            f"frozen: {frozen:,}  total: {total:,}",
            flush=True,
        )
        return trainable, frozen, total

    def get_multiple_latent_predictions(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
        num_samples: int = 10,
        z_goal=None,
    ) -> torch.Tensor:
        """
        ``num_samples`` independent draws: posterior MC if ``z_goal`` is set,
        else prior samples p(z|x, text) for best-of-K planning.
        """
        self.eval()
        with torch.no_grad():
            img_feat = self.backbone(x)
        text_feat = self.text_encoder(text_tokens)
        cond = torch.cat([img_feat, text_feat], dim=1)

        prior_mu = self.prior_mu(cond)
        prior_logvar = self.prior_logvar(cond)

        if z_goal is not None:
            mu, logvar = self.encode(img_feat, text_feat, z_goal)
        else:
            mu, logvar = prior_mu, prior_logvar

        preds = []
        for _ in range(num_samples):
            z = self.reparameterize(mu, logvar)
            z_pred = self.decoder(z)
            preds.append(z_pred)
        return torch.stack(preds, dim=0)
