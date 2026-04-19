"""Evaluation and visualization for CUB-200 linear probing runs."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


def load_feature_cache(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    return data["features"], data["labels"]


def load_cub_class_names(data_root: Path) -> List[str]:
    classes_path = data_root / "classes.txt"
    class_map: Dict[int, str] = {}

    with open(classes_path, "r", encoding="utf-8") as f:
        for line in f:
            idx_str, name = line.strip().split(maxsplit=1)
            class_map[int(idx_str) - 1] = name

    return [class_map[i] for i in range(len(class_map))]


def strip_numeric_prefix(class_name: str) -> str:
    short = re.sub(r"^\d+\.", "", class_name)
    return short.replace("_", " ")


def topk_accuracy_from_logits(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, logits.shape[1])
    topk_idx = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    correct = (topk_idx == labels[:, None]).any(axis=1)
    return float(correct.mean() * 100.0)


def compute_predictions_sklearn(
    model_path: Path,
    test_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load sklearn model and compute predictions."""
    clf = joblib.load(str(model_path))
    logits = clf.decision_function(test_features) if hasattr(clf, 'decision_function') else clf.predict_proba(test_features)
    pred_top1 = clf.predict(test_features)
    return logits, pred_top1


def save_confusion_ranked_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    short_class_names: List[str],
    figures_dir: Path,
) -> None:
    num_classes = len(short_class_names)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # Keep full matrix for potential downstream analysis.
    np.save(figures_dir / "confusion_matrix_full.npy", cm)

    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_no_diag[i, j] > 0:
                confused_pairs.append((i, j, int(cm_no_diag[i, j])))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = confused_pairs[:15]

    if not top_pairs:
        print("No non-diagonal confusion entries found; skipping confusion heatmap.")
        return

    true_ids = []
    pred_ids = []
    for t, p, _ in top_pairs:
        if t not in true_ids:
            true_ids.append(t)
        if p not in pred_ids:
            pred_ids.append(p)

    heat = np.zeros((len(true_ids), len(pred_ids)), dtype=np.int32)
    true_idx = {c: i for i, c in enumerate(true_ids)}
    pred_idx = {c: i for i, c in enumerate(pred_ids)}

    for t, p, count in top_pairs:
        heat[true_idx[t], pred_idx[p]] = count

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        heat,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=[short_class_names[i] for i in pred_ids],
        yticklabels=[short_class_names[i] for i in true_ids],
        cbar_kws={"label": "Confusion Count"},
    )
    plt.title("Top 15 Most Confused Class Pairs")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusions_top15_heatmap.png", dpi=150)
    plt.close()


def balanced_subsample_indices(labels: np.ndarray, total_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, total_samples // len(classes))

    selected: List[int] = []
    leftovers: List[np.ndarray] = []

    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        take = min(per_class, len(idx))
        selected.extend(idx[:take].tolist())
        if len(idx) > take:
            leftovers.append(idx[take:])

    remaining = total_samples - len(selected)
    if remaining > 0 and leftovers:
        pool = np.concatenate(leftovers)
        rng.shuffle(pool)
        selected.extend(pool[:remaining].tolist())

    selected = np.array(selected, dtype=np.int64)
    if len(selected) > total_samples:
        rng.shuffle(selected)
        selected = selected[:total_samples]
    return selected


def build_order_groups(
    labels: np.ndarray,
    class_names: List[str],
    order_map_path: Optional[Path],
) -> Tuple[np.ndarray, List[str], str]:
    if order_map_path is None or not order_map_path.exists():
        return labels, [strip_numeric_prefix(name) for name in class_names], "class"

    with open(order_map_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)

    class_to_group = {}
    for idx, name in enumerate(class_names):
        short_name = strip_numeric_prefix(name)
        mapped = raw_map.get(str(idx))
        if mapped is None:
            mapped = raw_map.get(short_name)
        class_to_group[idx] = mapped if mapped is not None else short_name

    groups = sorted(set(class_to_group.values()))
    group_to_id = {g: i for i, g in enumerate(groups)}
    mapped_labels = np.array([group_to_id[class_to_group[int(c)]] for c in labels], dtype=np.int64)
    return mapped_labels, groups, "order"


def save_tsne_plot(
    test_features: np.ndarray,
    test_labels: np.ndarray,
    class_names: List[str],
    figures_dir: Path,
    color_mode: str,
    order_map_path: Optional[Path],
    tsne_samples: int,
    seed: int,
) -> None:
    idx = balanced_subsample_indices(test_labels, total_samples=tsne_samples, seed=seed)
    x_sub = test_features[idx]
    y_sub = test_labels[idx]

    if color_mode == "order":
        color_labels, legend_names, effective_mode = build_order_groups(y_sub, class_names, order_map_path)
        if effective_mode != "order":
            print("Order map missing/unavailable; falling back to class coloring for t-SNE.")
            color_labels = y_sub
            legend_names = [strip_numeric_prefix(name) for name in class_names]
    else:
        color_labels = y_sub
        legend_names = [strip_numeric_prefix(name) for name in class_names]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(x_sub)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=color_labels,
        cmap="tab20",
        s=10,
        alpha=0.85,
        linewidths=0,
    )

    plt.title(f"t-SNE of Test Features ({len(idx)} Samples)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Legend is only feasible when color groups are reasonably small.
    unique_groups = np.unique(color_labels)
    if len(unique_groups) <= 25:
        handles = []
        for g in unique_groups:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=legend_names[int(g)] if int(g) < len(legend_names) else f"Group {int(g)}",
                    markerfacecolor=scatter.cmap(scatter.norm(g)),
                    markersize=6,
                )
            )
        plt.legend(handles=handles, loc="best", fontsize=7, frameon=True)

    plt.tight_layout()
    plt.savefig(figures_dir / "tsne_test_features.png", dpi=150)
    plt.close()


def save_lowest_per_class_accuracy_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    short_class_names: List[str],
    figures_dir: Path,
) -> None:
    num_classes = len(short_class_names)
    per_class_acc = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        idx = np.where(y_true == c)[0]
        if len(idx) == 0:
            per_class_acc[c] = 0.0
        else:
            per_class_acc[c] = float((y_pred[idx] == y_true[idx]).mean() * 100.0)

    worst_idx = np.argsort(per_class_acc)[:20]

    plt.figure(figsize=(12, 8))
    plt.barh(
        [short_class_names[i] for i in worst_idx],
        per_class_acc[worst_idx],
        color="#d55e00",
    )
    plt.xlabel("Per-Class Top-1 Accuracy (%)")
    plt.ylabel("Class")
    plt.title("20 Lowest-Accuracy Classes")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(figures_dir / "per_class_lowest20.png", dpi=150)
    plt.close()


def update_results_json(results_path: Path, top1: float, top5: float, extra: Dict) -> None:
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    results["test_top1"] = top1
    results["test_top5"] = top5
    results["evaluation"] = extra

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate linear probe run and generate report figures.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to CUB_200_2011 root (for class names).")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing sklearn model and results.json.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory containing cached feature .npz files.")
    parser.add_argument("--model", type=str, default=None, help="Optional explicit model path.")
    parser.add_argument("--tsne_color_mode", type=str, default="class", choices=["class", "order"], help="Color t-SNE points by class or order mapping.")
    parser.add_argument("--order_map_json", type=str, default=None, help="Optional JSON mapping for order grouping.")
    parser.add_argument("--tsne_samples", type=int, default=1500, help="Balanced test samples for t-SNE.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and t-SNE.")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    run_dir = Path(args.run_dir)
    cache_dir = Path(args.cache_dir)

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    test_npz = cache_dir / "test_features.npz"
    model_path = Path(args.model) if args.model else run_dir / "sklearn_classifier.pkl"
    results_path = run_dir / "results.json"
    order_map_path = Path(args.order_map_json) if args.order_map_json else None

    if not test_npz.exists():
        raise FileNotFoundError(f"Test feature cache not found: {test_npz}")
    if not model_path.exists():
        raise FileNotFoundError(f"Sklearn model not found: {model_path}")

    class_names = load_cub_class_names(data_root)
    short_names = [strip_numeric_prefix(name) for name in class_names]

    x_test, y_test = load_feature_cache(test_npz)
    logits, y_pred = compute_predictions_sklearn(model_path, x_test)

    test_top1 = topk_accuracy_from_logits(logits, y_test, k=1)
    test_top5 = topk_accuracy_from_logits(logits, y_test, k=5)

    print(f"Test Top-1: {test_top1:.2f}%")
    print(f"Test Top-5: {test_top5:.2f}%")

    save_confusion_ranked_heatmap(y_test, y_pred, short_names, figures_dir)
    save_tsne_plot(
        test_features=x_test,
        test_labels=y_test,
        class_names=class_names,
        figures_dir=figures_dir,
        color_mode=args.tsne_color_mode,
        order_map_path=order_map_path,
        tsne_samples=args.tsne_samples,
        seed=args.seed,
    )
    save_lowest_per_class_accuracy_plot(y_test, y_pred, short_names, figures_dir)

    update_results_json(
        results_path,
        top1=test_top1,
        top5=test_top5,
        extra={
            "model": str(model_path),
            "test_npz": str(test_npz),
            "figures_dir": str(figures_dir),
            "tsne_color_mode": args.tsne_color_mode,
            "tsne_samples": args.tsne_samples,
            "seed": args.seed,
        },
    )

    print(f"Saved figures to: {figures_dir}")
    print(f"Updated results file: {results_path}")


if __name__ == "__main__":
    main()
