#!/usr/bin/env python3
"""
Train a model to match the NTField **goal-side latent** (the row fed into the
metric head) using **text prompt**, **RGB image**, and **start joint configuration**.

The frozen trajectory NTField defines the teacher:
  z_start, z_goal = network.encode_pair_latents( concat(q_start, q_goal) )

The student predicts z_goal_hat ≈ z_goal from (prompt, image, q_start), so at
inference you can replace an explicit goal configuration with the predicted
latent when building planner inputs (requires a small planner change).

Dataset: ``grasp_6dof_demo_*.h5`` from collect_data (``images``, ``joint_configs``,
``final_joint_config``, attrs ``prompt``).

Usage:
  cd hanwen_grasping
  python train_goal_rep_alignment.py \\
    --checkpoint ../ntrl-demo/Experiments/UR5_trajectory/.../Model_Epoch_04300_ValLoss_*.pt \\
    --h5_glob '../collected_data/grasp_6dof_demo_*.h5' \\
    --epochs 20 --batch_size 16 --out goal_rep_student.pt

Requires: torch, h5py, numpy; torchvision recommended for image resize.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision.models as models
from transformers import DistilBertModel, DistilBertTokenizer

# NTField input normalization (same as planning/gradient_planner_trajectory.py)
SCALE = float(np.pi / 0.5)

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PI_VLA_ROOT = os.path.dirname(_FILE_DIR)
_NTRL_DEMO = os.path.join(_PI_VLA_ROOT, "ntrl-demo")
if os.path.isdir(_NTRL_DEMO) and _NTRL_DEMO not in sys.path:
    sys.path.insert(0, _NTRL_DEMO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _maybe_torchvision():
    try:
        import torchvision.transforms as T  # noqa: WPS433

        return T
    except ImportError:
        return None


def normalize_coords_tensor(q6: torch.Tensor, use_scale: bool) -> torch.Tensor:
    if use_scale:
        return q6 / SCALE
    return q6


def build_coords_batch(
    q_start: torch.Tensor, q_goal: torch.Tensor, use_scale: bool
) -> torch.Tensor:
    """q_start, q_goal: (B, 6) radians -> (B, 12) teacher input."""
    if use_scale:
        q_start = q_start / SCALE
        q_goal = q_goal / SCALE
    return torch.cat([q_start, q_goal], dim=1)

# ==========================================
# 1. Frozen Pre-Trained Image Encoder
# ==========================================
class PretrainedImageEncoder(nn.Module):
    """
    Acts as a proxy for R3M / VC-1. 
    Uses a pre-trained ResNet, frozen to prevent downstream catastrophic forgetting.
    """
    def __init__(self, out_dim: int = 2048):
        super().__init__()
        # Load a pre-trained ResNet50 (Standard practice for R3M base)
        # Note: If using the actual r3m library, you would load it here:
        # self.backbone = r3m.load_model("resnet50")
        # torchvision compatibility:
        # - newer: resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # - older: resnet50(pretrained=True)
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            resnet = models.resnet50(pretrained=True)
        
        # Strip the final classification layer to get the raw feature vector
        # and keep access to named stages so we can selectively unfreeze layer4.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = out_dim
        
        # Freeze all visual backbone parameters first.
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Then unfreeze only the last residual stage (layer4) to adapt to
        # overhead grasping-domain imagery while keeping earlier features stable.
        # In torchvision ResNet child ordering, index 7 corresponds to layer4.
        for param in self.backbone[7].parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No no_grad wrapper — layer4 needs to receive gradients.
        # Frozen layers won't update anyway since requires_grad=False.
        features = self.backbone(x)
        return features.flatten(1)


# ==========================================
# 2. Frozen Pre-Trained Text Encoder
# ==========================================
class PretrainedTextEncoder(nn.Module):
    """
    Extracts rich language embeddings using a pre-trained Transformer.
    """
    def __init__(self, model_name: str = 'distilbert-base-uncased', out_dim: int = 768):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        # Older torch versions (e.g., 1.x) can fail loading safetensors
        # due to missing torch.frombuffer; force .bin checkpoint loading.
        self.transformer = DistilBertModel.from_pretrained(
            model_name,
            use_safetensors=False,
        )
        self.out_dim = out_dim
        
        # FREEZE the language backbone
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        # Tokenize and pad the batch of text prompts
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=64
        ).to(device)
        
        with torch.no_grad():
            outputs = self.transformer(**inputs)
            
        # Extract the [CLS] token representation as the sentence embedding
        sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding # Output shape: (B, 768)


# ==========================================
# 3. FiLM (Feature-wise Linear Modulation)
# ==========================================
class FiLMConditioning(nn.Module):
    """
    Injects language embeddings into the visual representation by learning 
    scale (gamma) and shift (beta) parameters from the text.
    """
    def __init__(self, text_dim: int, image_dim: int):
        super().__init__()
        # Maps the text embedding to the exact dimensions of the image embedding
        # Output is 2 * image_dim because we need both gamma and beta
        self.film_generator = nn.Linear(text_dim, image_dim * 2)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        # Generate conditioning parameters from text
        film_params = self.film_generator(text_features)
        
        # Split into scale (gamma) and shift (beta)
        # Note: We add 1 to gamma so that the default scale (before learning) is 1, not 0
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma + 1.0 
        
        # Apply FiLM: condition the image features using the text parameters
        conditioned_features = (image_features * gamma) + beta
        return conditioned_features


# ==========================================
# 4. The Updated Student Model
# ==========================================
class GoalLatentPredictorWithFiLM(nn.Module):
    """
    Predicts NTField goal latent using frozen Foundation Models and FiLM.
    """
    def __init__(self, ntfield_h: int = 256):
        super().__init__()
        
        # 1. Initialize Frozen Encoders
        self.img_enc = PretrainedImageEncoder(out_dim=2048)
        self.text_enc = PretrainedTextEncoder(out_dim=768)
        
        # 2. Initialize FiLM Fusion Module
        self.film = FiLMConditioning(text_dim=self.text_enc.out_dim, image_dim=self.img_enc.out_dim)
        
        # 3. Start Joint Encoder (q_start) - actively trained
        self.q_enc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        
        # 4. Final Policy/Metric Head - actively trained
        # Input = Conditioned Image Features (2048) + Encoded Joints (128)
        fused_dim = self.img_enc.out_dim + 128
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, ntfield_h),
        )

    def forward(self, images_bchw: torch.Tensor, prompts: list[str], q_start: torch.Tensor) -> torch.Tensor:
        # Device management
        device = images_bchw.device
        
        # 1. Extract Frozen Features
        im_feats = self.img_enc(images_bchw)
        text_feats = self.text_enc(prompts, device)
        
        # 2. Fuse Vision and Language via FiLM
        # The visual features are actively shifted and scaled based on the text instruction
        vis_lang_fused = self.film(im_feats, text_feats)
        
        # 3. Encode Proprioception (Joints)
        q = self.q_enc(q_start)
        
        # 4. Predict the Goal Latent
        final_input = torch.cat([vis_lang_fused, q], dim=1)
        z_goal_hat = self.head(final_input)
        
        return z_goal_hat

class H5GraspDemoDataset(Dataset):
    """One sample = (image_i, prompt, q_start_i, q_goal_final)."""

    def __init__(
        self,
        h5_paths: List[str],
        image_key: str = "images",
        use_torchvision_resize: bool = True,
        img_size: int = 128,
    ):
        self.samples: List[Tuple[str, int]] = []
        self.prompts_per_file: Dict[str, str] = {}
        self._bad_samples: set[Tuple[str, int]] = set()
        self._bad_files: set[str] = set()
        self._bad_file_hits: Dict[str, int] = {}
        self.image_key = image_key
        self.img_size = img_size
        self._T = _maybe_torchvision()
        self._use_tv = bool(use_torchvision_resize and self._T is not None)
        if self._use_tv:
            self._tfm = self._T.Compose(
                [
                    self._T.ToPILImage(),
                    self._T.Resize((224, 224)), # Standard ResNet size
                    self._T.ToTensor(),
                    self._T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # CRITICAL
                ]
            )
        else:
            self._tfm = None

        import h5py

        for path in h5_paths:
            path = os.path.abspath(path)
            try:
                with h5py.File(path, "r") as f:
                    if image_key not in f or "joint_configs" not in f or "final_joint_config" not in f:
                        continue
                    n = f[image_key].shape[0]
                    pr = f.attrs.get("prompt", "")
                    if isinstance(pr, bytes):
                        pr = pr.decode("utf-8", errors="replace")
                    self.prompts_per_file[path] = str(pr)
                    for i in range(n):
                        self.samples.append((path, i))
            except OSError:
                print(f"[warn] Skipping non-HDF5 or corrupted file: {path}")
                continue

        if not self.samples:
            raise ValueError("No samples found. Check --h5_glob and HDF5 keys (images, joint_configs).")

    def __len__(self) -> int:
        return len(self.samples)

    def _image_to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if self._use_tv:
            return self._tfm(rgb)
        # numpy fallback: mean-pool downsample
        h, w = rgb.shape[:2]
        tgt = self.img_size
        if h != tgt or w != tgt:
            ys = (np.linspace(0, h - 1, tgt)).astype(int)
            xs = (np.linspace(0, w - 1, tgt)).astype(int)
            rgb = rgb[np.ix_(ys, xs)]
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t

    def __getitem__(self, idx: int):
        import h5py

        n = len(self.samples)
        max_retries = min(256, n)
        # Probe varied indices to avoid getting stuck in one corrupted region/file.
        for step in range(max_retries):
            # Large odd stride gives good coverage across the dataset index space.
            j = (idx + step * 9973) % n
            path, i = self.samples[j]
            if path in self._bad_files or (path, i) in self._bad_samples:
                continue
            try:
                with h5py.File(path, "r") as f:
                    img = np.array(f[self.image_key][i])
                    q_start = np.array(f["joint_configs"][i, :6], dtype=np.float32)
                    q_goal = np.array(f["final_joint_config"][:6], dtype=np.float32)
                pr = self.prompts_per_file[path]
                x = self._image_to_tensor(img)
                return x, pr, torch.from_numpy(q_start), torch.from_numpy(q_goal)
            except OSError:
                self._bad_samples.add((path, i))
                hits = self._bad_file_hits.get(path, 0) + 1
                self._bad_file_hits[path] = hits
                print(f"[warn] Skipping unreadable sample: file={path} idx={i}")
                if hits >= 32:
                    self._bad_files.add(path)
                    print(f"[warn] Marking file as bad after repeated read failures: {path}")
                continue

        raise RuntimeError(
            f"Failed to read dataset sample at idx={idx}; no readable sample found after {max_retries} attempts."
        )


def collate_fn(batch):
    imgs, prs, qs, qg = zip(*batch)
    return torch.stack(imgs, 0), list(prs), torch.stack(qs, 0), torch.stack(qg, 0)


def load_teacher(
    checkpoint: str, device: torch.device, data_path: Optional[str] = None
) -> Tuple[torch.nn.Module, int]:
    from models.metric_arm import model_test_metric as md

    model_path = os.path.dirname(os.path.abspath(checkpoint))
    if data_path is None:
        data_path = os.path.join(_NTRL_DEMO, "datasets", "arm", "UR5_trajectory")
    model = md.Model(model_path, data_path, dim=6, source=[0.0] * 6, device=str(device))
    model.load(checkpoint)
    model.network.eval()
    for p in model.network.parameters():
        p.requires_grad_(False)
    h = 256
    # Infer H from first linear after encoder stack
    if hasattr(model.network, "encoder") and len(model.network.encoder) > 0:
        lin = model.network.encoder[-1]
        if hasattr(lin, "out_features"):
            h = int(lin.out_features)
    return model.network, h


def main() -> None:
    p = argparse.ArgumentParser(description="Train goal latent alignment (prompt+image+q_start -> z_goal)")
    p.add_argument("--checkpoint", type=str, required=True, help="NTField Model_Epoch_*.pt")
    p.add_argument("--h5_glob", type=str, required=True, help="Glob for grasp_6dof_demo_*.h5")
    p.add_argument("--out", type=str, default="goal_rep_student.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--warmup_pct",
        type=float,
        default=0.1,
        help="Fraction of total OneCycle steps used for linear warmup.",
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_key", type=str, default="images", help="HDF5 dataset name (images or image)")
    p.add_argument(
        "--normalize_coords",
        action="store_true",
        help="Divide joints by π/0.5 before NTField (matches gradient_planner_trajectory). "
        "Try without first if loss is unstable; trajectory points.npy is often raw radians.",
    )
    p.add_argument(
        "--loss",
        type=str,
        choices=["mse", "cosine"],
        default="mse",
        help="cosine = 1 - cos(z_hat, z_goal); useful if scales drift",
    )
    p.add_argument(
        "--triplet_weight",
        type=float,
        default=0.1,
        help="Weight for batch triplet margin loss added to base loss.",
    )
    p.add_argument(
        "--triplet_margin",
        type=float,
        default=0.5,
        help="Margin used by batch triplet loss (anchor=z_hat, pos=z_goal_i, neg=z_goal_j).",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    paths = sorted(glob.glob(args.h5_glob))
    paths = [os.path.abspath(x) for x in paths if os.path.isfile(x)]
    if not paths:
        print(f"No HDF5 matched: {args.h5_glob}")
        sys.exit(1)
    print(f"Using {len(paths)} HDF5 files")

    ds_full = H5GraspDemoDataset(paths, image_key=args.image_key)
    print(f"Samples {len(ds_full)}")

    n = len(ds_full)
    n_val = max(1, int(0.1 * n))
    indices = np.random.RandomState(0).permutation(n)
    val_list = sorted(indices[:n_val].tolist())
    val_set = set(val_list)
    train_idx = [i for i in range(n) if i not in val_set]

    ds_tr = Subset(ds_full, train_idx)
    ds_va = Subset(ds_full, val_list)

    loader_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    teacher, nt_h = load_teacher(args.checkpoint, device)
    student = GoalLatentPredictorWithFiLM(ntfield_h=nt_h).to(device)
    # Only pass parameters that actually require gradients
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
        steps_per_epoch=len(loader_tr),
        epochs=args.epochs,
        pct_start=args.warmup_pct,
    )

    def teacher_z_goal(qs: torch.Tensor, qg: torch.Tensor) -> torch.Tensor:
        coords = build_coords_batch(qs, qg, args.normalize_coords).to(device)
        with torch.no_grad():
            _, zg = teacher.encode_pair_latents(coords)
        return zg

    def base_alignment_loss(z_hat: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        if args.loss == "mse":
            return F.mse_loss(z_hat, z_tgt)
        z_hat_n = F.normalize(z_hat, dim=1)
        z_tgt_n = F.normalize(z_tgt, dim=1)
        return (1.0 - (z_hat_n * z_tgt_n).sum(dim=1)).mean()

    def batch_triplet_loss(z_hat: torch.Tensor, z_tgt: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
        if z_hat.shape[0] <= 1:
            return z_hat.new_tensor(0.0)

        # Normalize FIRST — margin is meaningful in unit-sphere geometry.
        z_hat = F.normalize(z_hat, dim=1)
        z_tgt = F.normalize(z_tgt, dim=1)

        # Hard negatives: closest distinct target in the batch.
        # Mask out near-identical vectors (including the diagonal and duplicates)
        # to avoid false negatives that produce uninformative triplet gradients.
        with torch.no_grad():
            dists = torch.cdist(z_tgt, z_tgt, p=2)
            dists[dists < 1e-5] = float("inf")
            if torch.isinf(dists).all():
                return z_hat.new_tensor(0.0)
            neg_idx = dists.argmin(dim=1)

        neg_z = z_tgt[neg_idx]

        return F.triplet_margin_loss(z_hat, z_tgt, neg_z, margin=margin)

    out_path = os.path.abspath(args.out)

    for epoch in range(args.epochs):
        student.train()
        run = 0.0
        n_b = 0
        for imgs, prs, qs, qg in loader_tr:
            imgs = imgs.to(device)
            qs = qs.to(device)
            qg = qg.to(device)
            z_tgt = teacher_z_goal(qs, qg)
            z_hat = student(imgs, prs, qs)
            align_loss = base_alignment_loss(z_hat, z_tgt)
            trip_loss = batch_triplet_loss(z_hat, z_tgt, margin=args.triplet_margin)
            loss = align_loss + args.triplet_weight * trip_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            run += float(loss.item())
            n_b += 1
        train_loss = run / max(1, n_b)

        student.eval()
        vrun = 0.0
        vb = 0
        with torch.no_grad():
            for imgs, prs, qs, qg in loader_va:
                imgs = imgs.to(device)
                qs = qs.to(device)
                qg = qg.to(device)
                z_tgt = teacher_z_goal(qs, qg)
                z_hat = student(imgs, prs, qs)
                align_loss = base_alignment_loss(z_hat, z_tgt)
                trip_loss = batch_triplet_loss(z_hat, z_tgt, margin=args.triplet_margin)
                loss = align_loss + args.triplet_weight * trip_loss
                vrun += float(loss.item())
                vb += 1
        val_loss = vrun / max(1, vb)
        print(
            f"epoch {epoch+1}/{args.epochs}  train_{args.loss}={train_loss:.6f}  val_{args.loss}={val_loss:.6f}",
            flush=True,
        )

        # Save checkpoint every epoch with an epoch-specific suffix.
        epoch_suffix = f"_epoch{epoch+1:03d}"
        base, ext = os.path.splitext(out_path)
        epoch_path = base + epoch_suffix + (ext or ".pt")
        payload_epoch = {
            "student_state_dict": student.state_dict(),
            "ntfield_h": nt_h,
            "normalize_coords": args.normalize_coords,
            "loss": args.loss,
            "triplet_weight": args.triplet_weight,
            "triplet_margin": args.triplet_margin,
            "lr_scheduler": "OneCycleLR",
            "warmup_pct": args.warmup_pct,
            "image_key": args.image_key,
            "checkpoint_teacher": os.path.abspath(args.checkpoint),
            "epoch": int(epoch + 1),
        }
        torch.save(payload_epoch, epoch_path)
        with open(epoch_path + ".json", "w") as f:
            json.dump({k: v for k, v in payload_epoch.items() if k != "student_state_dict"}, f, indent=2)

    # Final checkpoint at the user-specified output path (no epoch suffix).
    payload_final = {
        "student_state_dict": student.state_dict(),
        "ntfield_h": nt_h,
        "normalize_coords": args.normalize_coords,
        "loss": args.loss,
        "triplet_weight": args.triplet_weight,
        "triplet_margin": args.triplet_margin,
        "lr_scheduler": "OneCycleLR",
        "warmup_pct": args.warmup_pct,
        "image_key": args.image_key,
        "checkpoint_teacher": os.path.abspath(args.checkpoint),
        "epoch": int(args.epochs),
    }
    torch.save(payload_final, out_path)
    with open(out_path + ".json", "w") as f:
        json.dump({k: v for k, v in payload_final.items() if k != "student_state_dict"}, f, indent=2)
    print(f"Saved final student to {out_path}", flush=True)


if __name__ == "__main__":
    main()
