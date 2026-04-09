# Image → object location (`img2objloc_model`)

Pipeline: HDF5 demos → Segment Anything table crop → train a small CNN regressor for **(x, y)** with **z fixed** at `0.1` (aligned with Isaac table `table_dims.z`), then evaluate on a held-out test folder.

## SAM checkpoint (ViT-H)

The Segment Anything ViT-H weights are **not** in this repo (too large for GitHub). Download `sam_vit_h_4b8939.pth` into `img2objloc_model/checkpoints/` from the repo root (`PI-VLA/`):

```bash
mkdir -p img2objloc_model/checkpoints
wget -O img2objloc_model/checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Or with curl:

```bash
mkdir -p img2objloc_model/checkpoints
curl -L -o img2objloc_model/checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Use `--sam-checkpoint img2objloc_model/checkpoints/sam_vit_h_4b8939.pth` in the scripts below.

## Data layout

- Raw demos: `output/data_collection/20260402/*.h5`  
  Each file has `images` `(100, H, W, 3)`, `object_location` `(3,)`, etc. We use **`images[0]`** and labels from **`object_location`**, with **z overwritten to 0.1** in exported metadata (override with `--label-z` on SAM export).

- SAM exports: `output/segment_anything/<timestamp>/` (one directory per episode), containing at least:
  - `first_image_table_crop.png` — cropped RGB around the table (**object stays visible**)
  - `object_location.json` — `object_location_original`, `object_location_z_fixed` (x, y, 0.1 by default)

- Train / test split (80/20): after moving samples with `split_segment_anything_train_test.py`:
  - `output/segment_anything/train/<timestamp>/`
  - `output/segment_anything/test/<timestamp>/`

## Scripts

### 1. Table crop with Segment Anything (single or batch)

Requires: `segment-anything` + a SAM checkpoint (e.g. `sam_vit_h_4b8939.pth`).

```bash
# Single file
python img2objloc_model/extract_table_with_sam.py \
  --h5-path output/data_collection/20260402/grasp_6dof_demo_20260402_143557.h5 \
  --sam-checkpoint img2objloc_model/checkpoints/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --output-dir output/segment_anything

# All H5 in a directory (writes <timestamp>/ under output-dir)
python img2objloc_model/extract_table_with_sam.py \
  --h5-dir output/data_collection/20260402 \
  --sam-checkpoint img2objloc_model/checkpoints/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --output-dir output/segment_anything
```

Use **`first_image_table_crop.png`** for training (full pixels inside crop; not the blacked-out `table_only` mask).

### 2. Train / test directory split (8 : 2)

```bash
python img2objloc_model/split_segment_anything_train_test.py \
  --segment-root output/segment_anything \
  --train-ratio 0.8 \
  --seed 42
```

This **moves** timestamp folders into `output/segment_anything/train/` and `output/segment_anything/test/`.

### 3. Train (x, y) regressor

From the repo root (`PI-VLA/`):

```bash
python img2objloc_model/train_objloc_xy.py \
  --segment-root output/segment_anything/train \
  --image-name first_image_table_crop.png \
  --save-dir img2objloc_model/checkpoints_xy \
  --epochs 40 \
  --backbone resnet50 \
  --batch-size 32
```

**Shared GPU / low VRAM:** use AMP (on by default on CUDA), smaller `--batch-size`, and `--grad-accum-steps` to keep an effective large batch, e.g.:

```bash
python img2objloc_model/train_objloc_xy.py \
  --segment-root output/segment_anything/train \
  --backbone resnet50 \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --eval-batch-size 8
```

Optional: `--device cuda` or `CUDA_VISIBLE_DEVICES=1` to pick a GPU with free memory.

**Example run (validation metrics):**

```
[033/40] train_loss=0.000727 val_mae_xy=0.003845 val_rmse_xy=0.005040
  -> saved best model to img2objloc_model/checkpoints_xy/best_xy_model.pt
```

Best checkpoint: **`img2objloc_model/checkpoints_xy/best_xy_model.pt`**.

### 4. Evaluate on the test directory

```bash
python img2objloc_model/eval_objloc_xy.py \
  --segment-root output/segment_anything/test \
  --checkpoint img2objloc_model/checkpoints_xy/best_xy_model.pt \
  --batch-size 8
```

Optional CSV of per-episode predictions:

```bash
python img2objloc_model/eval_objloc_xy.py \
  --segment-root output/segment_anything/test \
  --predictions-csv img2objloc_model/test_predictions.csv
```

The model predicts **only x and y**; **`z_fixed`** in the checkpoint (default `0.1`, set with `train_objloc_xy.py --z-fixed`) is for downstream use, not learned.

## Dependencies

- Training: `torch`, `torchvision`, `Pillow`, `numpy`
- SAM export: `segment_anything`, OpenCV, `h5py`, SAM `.pth` weights

---

*On a multi-GPU node, check `nvidia-smi` and pick a device with enough free memory before starting long training runs.*
