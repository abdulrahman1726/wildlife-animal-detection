"""
train.py
--------
Train a YOLOv11 model on the Kaggle "Object Detection - Wildlife Dataset - YOLO Format"
(ankanghosh651/object-detection-wildlife-dataset-yolo-format).

Classes: buffalo, elephant, rhino, zebra

Usage:
    python train.py
"""

import os
import shutil
import kagglehub
from ultralytics import YOLO


def download_dataset() -> str:
    """Download the dataset from Kaggle using kagglehub and return its local path."""
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download(
        "ankanghosh651/object-detection-wildlife-dataset-yolo-format"
    )
    print(f"Dataset downloaded to: {path}")
    return path


def locate_data_yaml(dataset_root: str) -> str:
    """
    Find the data.yaml file inside the downloaded dataset.
    Kaggle YOLO-format datasets usually ship with a data.yaml already,
    but paths inside it point to the original author's machine, so we
    rewrite it to point to the actual local folders.
    """
    yaml_path = None
    for root, _, files in os.walk(dataset_root):
        for f in files:
            if f == "data.yaml":
                yaml_path = os.path.join(root, f)
                break
        if yaml_path:
            break

    if yaml_path is None:
        raise FileNotFoundError(
            "Could not find data.yaml inside the downloaded dataset. "
            "Inspect the dataset folder structure manually."
        )

    dataset_dir = os.path.dirname(yaml_path)
    fixed_yaml_path = os.path.join(dataset_dir, "data_fixed.yaml")

    # Rebuild a clean data.yaml with absolute local paths
    train_dir = os.path.join(dataset_dir, "train", "images")
    val_dir = os.path.join(dataset_dir, "valid", "images")
    test_dir = os.path.join(dataset_dir, "test", "images")

    # Fall back gracefully if folder names differ slightly (val vs valid)
    if not os.path.isdir(val_dir):
        alt = os.path.join(dataset_dir, "val", "images")
        if os.path.isdir(alt):
            val_dir = alt

    content = f"""train: {train_dir}
val: {val_dir}
test: {test_dir if os.path.isdir(test_dir) else val_dir}

nc: 4
names: ['buffalo', 'elephant', 'rhino', 'zebra']
"""
    with open(fixed_yaml_path, "w") as f:
        f.write(content)

    print(f"Rewritten data.yaml saved to: {fixed_yaml_path}")
    print(content)
    return fixed_yaml_path


def train_model(data_yaml: str, epochs: int = 100, imgsz: int = 640, model_size: str = "yolo11s.pt"):
    """
    Train YOLOv11 on the wildlife dataset.

    model_size options (smallest -> largest):
        yolo11n.pt  - nano   (fastest, least accurate)
        yolo11s.pt  - small  (good balance for ~1500 images) <-- recommended default
        yolo11m.pt  - medium
        yolo11l.pt  - large
        yolo11x.pt  - extra large
    """
    print(f"Loading base model: {model_size}")
    model = YOLO(model_size)

    print("Starting training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        patience=20,          # early stopping
        project="runs_wildlife",
        name="yolo11_wildlife",
        exist_ok=True,
        plots=True,
    )

    # Locate best weights
    best_weights = os.path.join("runs_wildlife", "yolo11_wildlife", "weights", "best.pt")
    print(f"\nTraining complete. Best weights at: {best_weights}")

    # Copy best weights to project root for easy access by the Streamlit app
    target = "best.pt"
    if os.path.exists(best_weights):
        shutil.copy(best_weights, target)
        print(f"Copied best weights to: {os.path.abspath(target)}")

    return results, best_weights


def evaluate_model(model_path: str, data_yaml: str):
    """Run validation metrics (mAP50, mAP50-95, precision, recall) on the test/val split."""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split="test")
    print(metrics)
    return metrics


if __name__ == "__main__":
    dataset_root = download_dataset()
    data_yaml_path = locate_data_yaml(dataset_root)

    # YOLOv11-small is a good default: dataset is only ~1500 images across 4 classes,
    # so nano/small give the best accuracy-to-training-time tradeoff without overfitting.
    _, best_weights_path = train_model(
        data_yaml=data_yaml_path,
        epochs=100,
        imgsz=640,
        model_size="yolo11s.pt",
    )

    if os.path.exists(best_weights_path):
        evaluate_model(best_weights_path, data_yaml_path)
