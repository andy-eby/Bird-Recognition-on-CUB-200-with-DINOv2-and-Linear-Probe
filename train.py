"""End-to-end training entrypoint for CUB-200 linear probing on cached DINOv2 features."""

import argparse
import json
import time
import warnings
from pathlib import Path

from scripts.extract_features import extract_features
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def ensure_feature_cache(
    data_root: str,
    cache_dir: str,
    model_variant: str,
    feature_batch_size: int,
    device: str,
) -> dict:
    """Ensure train/val/test feature caches exist; extract missing splits."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    split_to_file = {
        "train": cache_path / "train_features.npz",
        "val": cache_path / "val_features.npz",
        "test": cache_path / "test_features.npz",
    }

    missing_splits = [split for split, npz in split_to_file.items() if not npz.exists()]

    if not missing_splits:
        print("All cache files found. Skipping feature extraction.")
    else:
        print(f"Missing cache for splits: {missing_splits}")
        print("Running automatic feature extraction for missing splits...")
        for split in missing_splits:
            extract_features(
                data_root=data_root,
                split=split,
                model_variant=model_variant,
                batch_size=feature_batch_size,
                output_dir=str(cache_path),
                device=None if device == "auto" else device,
            )

    return {split: str(path) for split, path in split_to_file.items()}


def confirm_continue_on_low_accuracy(test_acc_c1: float, threshold: float = 75.0) -> None:
    """Warn and ask user confirmation if sklearn accuracy is low."""
    if test_acc_c1 >= threshold:
        return

    warnings.warn(
        f"Test accuracy is {test_acc_c1:.2f}% (< {threshold:.2f}%). "
        "This can indicate an upstream issue (normalization/token/split mismatch).",
        stacklevel=2,
    )

    while True:
        answer = input("Continue anyway? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            print("User confirmed continuation.")
            return
        if answer in ("", "n", "no"):
            raise RuntimeError("Aborted by user after low accuracy.")
        print("Please answer with 'y' or 'n'.")


def topk_accuracy_from_scores(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Compute Top-k accuracy from class scores/probabilities."""
    k = min(k, scores.shape[1])
    topk_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    correct = (topk_idx == labels[:, None]).any(axis=1)
    return float(correct.mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train linear probe on CUB-200 with DINOv2 cached features."
    )

    parser.add_argument("--data_root", type=str, required=True, help="Path to CUB_200_2011 root.")
    parser.add_argument("--model_variant", type=str, default="vitb14", choices=["vits14", "vitb14", "vitl14"], help="DINOv2 model variant.")
    parser.add_argument("--output_dir", type=str, default="./runs/cub_vitb14", help="Run output directory.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for cached feature .npz files.")
    parser.add_argument("--feature_batch_size", type=int, default=128, help="Batch size for feature extraction.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device.")
    parser.add_argument("--cv_folds", type=int, default=5, help="GridSearchCV folds for C hyperparameter sweep.")

    args = parser.parse_args()

    run_start = time.time()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1/3] Ensuring feature cache")
    cache_files = ensure_feature_cache(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        model_variant=args.model_variant,
        feature_batch_size=args.feature_batch_size,
        device=args.device,
    )

    print("\n[Step 2/3] Training scikit-learn classifier with GridSearchCV")
    
    # Load cached features
    train_data = np.load(cache_files["train"])
    test_data = np.load(cache_files["test"])
    
    train_features = train_data["features"]
    train_labels = train_data["labels"]
    test_features = test_data["features"]
    test_labels = test_data["labels"]
    
    # Create pipeline with feature scaling + logistic regression.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=1200,
            tol=1e-3,
            solver="lbfgs",
            multi_class="multinomial",
        ))
    ])
    
    # Focus on a small neighborhood around the known good C=10 to keep runtime short.
    c_values = [15.0, 17.5, 20.0, 22.5]
    
    # Use StratifiedKFold for better class distribution in folds
    cv_strategy = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    
    # Perform GridSearchCV for optimal C parameter
    clf = GridSearchCV(
        pipeline,
        param_grid={"classifier__C": c_values},
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1
    )
    
    clf.fit(train_features, train_labels)
    
    best_c = clf.best_params_["classifier__C"]
    test_accuracy = clf.score(test_features, test_labels)
    test_scores = (
        clf.decision_function(test_features)
        if hasattr(clf, "decision_function")
        else clf.predict_proba(test_features)
    )
    test_top5 = topk_accuracy_from_scores(test_scores, test_labels, k=5)
    
    print(f"Best C: {best_c}")
    print(f"Test Top-1 Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Top-5 Accuracy: {test_top5 * 100:.2f}%")
    
    confirm_continue_on_low_accuracy(test_accuracy * 100, threshold=75.0)

    total_seconds = time.time() - run_start

    # Save the trained sklearn model
    model_path = output_path / "sklearn_classifier.pkl"
    joblib.dump(clf, str(model_path))
    print(f"Saved sklearn classifier: {model_path}")

    print("\n[Step 3/3] Writing results.json")
    results = {
        "model_variant": args.model_variant,
        "dataset": "CUB-200-2011",
        "classifier": "sklearn-logreg-fast-tuned",
        "best_c": float(best_c),
        "test_top1": float(test_accuracy),
        "test_top5": float(test_top5),
        "total_training_time_sec": total_seconds,
        "hyperparameters": {
            "model": "Pipeline(StandardScaler -> LogisticRegression)",
            "max_iter": 1200,
            "tol": 1e-3,
            "solver": "lbfgs",
            "multi_class": "multinomial",
            "c_values": c_values,
            "cv_folds": args.cv_folds,
            "cv_strategy": "StratifiedKFold(shuffle=True, random_state=42)",
            "feature_batch_size": args.feature_batch_size,
            "device": args.device,
            "cache_dir": args.cache_dir,
            "output_dir": args.output_dir,
        },
        "grid_search_results": {
            "best_params": clf.best_params_,
            "best_score": float(clf.best_score_),
            "cv_results": {
                "mean_test_scores": [float(score) for score in clf.cv_results_["mean_test_score"]],
                "std_test_scores": [float(score) for score in clf.cv_results_["std_test_score"]],
                "params": clf.cv_results_["params"],
            }
        }
    }

    results_path = output_path / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results: {results_path}")
    print(f"Total elapsed time: {total_seconds:.2f}s")


if __name__ == "__main__":
    main()
