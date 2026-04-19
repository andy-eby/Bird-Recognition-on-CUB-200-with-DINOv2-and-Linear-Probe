# Fine-Grained Recognition on CUB-200 with DINOv2 + Scikit-Learn

This project implements a two-stage fine-grained recognition pipeline:

1. Extract frozen DINOv2 features once and cache them.
2. Train a scikit-learn LogisticRegression classifier with GridSearchCV on cached features.

The workflow is designed to be fast and efficient on CUB-200, leveraging pretrained DINOv2 features with a simple linear classifier.

## Project Structure

- `datasets/`
  - `cub200.py`: CUB-200 dataset parser with official split + stratified val split.
  - `transforms.py`: train/eval image transforms.
- `models/`
  - `dinov2.py`: frozen DINOv2 feature extractor.
- `scripts/`
  - `extract_features.py`: feature pre-extraction and caching.
- `train.py`: end-to-end pipeline (auto cache + sklearn training + results.json).
- `evaluate.py`: post-training metrics and figures.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Layout

Point `--data_root` to the CUB-200-2011 directory that contains:

- `images/`
- `images.txt`
- `image_class_labels.txt`
- `train_test_split.txt`
- `classes.txt`

Example:

```text
/data/CUB_200_2011
```

## Quick Start (Recommended)

### 1) Run End-to-End Training

This command will:
- check/create cached features for train/test,
- run scikit-learn GridSearchCV to find the optimal C parameter,
- save the trained classifier,
- save `results.json`.

```bash
python train.py --data_root data/CUB_200_2011 --model_variant vitb14 --output_dir ./runs/cub_vitb14
```

### 2) Run Evaluation + Figures

This command computes final metrics and writes report-ready figures to `./runs/cub_vitb14/figures/`.

```bash
python evaluate.py --data_root data/CUB_200_2011 --run_dir ./runs/cub_vitb14 --cache_dir ./cache
```

## Script-by-Script Usage

## A) Feature Extraction

Extract one split at a time:

```bash
python scripts/extract_features.py --data_root data/CUB_200_2011 --split train --model_variant vitb14 --batch_size 128 --output_dir ./cache
python scripts/extract_features.py --data_root data/CUB_200_2011 --split test  --model_variant vitb14 --batch_size 128 --output_dir ./cache
```

Outputs:
- `./cache/train_features.npz`
- `./cache/test_features.npz`

Each `.npz` contains:
- `features`: L2-normalized feature matrix
- `labels`: class IDs

## B) End-to-End Training

Main options:

- `--data_root`: CUB dataset root (required)
- `--model_variant`: `vits14`, `vitb14`, `vitl14` (default: `vitb14`)
- `--output_dir`: run directory for model, results (default: `./runs/cub_vitb14`)
- `--cache_dir`: cached features directory (default: `./cache`)
- `--feature_batch_size`: extraction batch size (default 128)
- `--device`: `auto`, `cpu`, `cuda` (default: `auto`)
- `--cv_folds`: GridSearchCV folds for C sweep (default: 5)

Example:

```bash
python train.py --data_root data/CUB_200_2011 --model_variant vitb14 --output_dir ./runs/cub_vitb14 --cv_folds 5
```

## C) Evaluation and Visualization

`evaluate.py` produces:

- Test Top-1 / Top-5 (printed and saved to results)
- Full confusion matrix (`confusion_matrix_full.npy`)
- Top-15 confused class-pair heatmap (`confusions_top15_heatmap.png`)
- t-SNE on balanced 1500 test samples (`tsne_test_features.png`)
- 20 lowest per-class accuracies (`per_class_lowest20.png`)

Default usage:

```bash
python evaluate.py --data_root data/CUB_200_2011 --run_dir ./runs/cub_vitb14 --cache_dir ./cache
```

Optional t-SNE coloring by bird order groups:

```bash
python evaluate.py --data_root data/CUB_200_2011 --run_dir ./runs/cub_vitb14 --cache_dir ./cache --tsne_color_mode order --order_map_json ./order_map.json
```

`order_map.json` can map either class index strings or short class names to an order/group label.

Example:

```json
{
  "0": "Procellariiformes",
  "1": "Passeriformes",
  "Black footed Albatross": "Procellariiformes"
}
```

## Output Files

Typical run directory contents:

```text
runs/cub_vitb14/
  sklearn_classifier.pkl
  results.json
  figures/
    confusion_matrix_full.npy
    confusions_top15_heatmap.png
    tsne_test_features.png
    per_class_lowest20.png
```

## Results Summary

After running `train.py` and `evaluate.py`, `results.json` will contain:

```json
{
  "model_variant": "vitb14",
  "dataset": "CUB-200-2011",
  "classifier": "sklearn-logreg",
  "best_c": 1.0,
  "test_top1": 0.85,
  "total_training_time_sec": 45.2,
  "hyperparameters": {
    "model": "LogisticRegression",
    "max_iter": 1000,
    "solver": "lbfgs",
    "multi_class": "multinomial",
    "c_values": [0.01, 0.1, 1.0, 10.0],
    "cv_folds": 5
  },
  "grid_search_results": {
    "best_params": {"C": 1.0},
    "best_score": 0.84,
    "cv_results": {...}
  }
}
```
    val_top1_curve.png
```

## Optional: View TensorBoard

```bash
tensorboard --logdir ./runs/cub_vitb14/tensorboard
```
