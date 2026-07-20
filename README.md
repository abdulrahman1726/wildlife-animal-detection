# Wildlife Animal Detection — YOLOv11 + Streamlit

Detects **buffalo, elephant, rhino, and zebra** in images/videos using a YOLOv11 model
trained on the Kaggle dataset
[`ankanghosh651/object-detection-wildlife-dataset-yolo-format`](https://www.kaggle.com/datasets/ankanghosh651/object-detection-wildlife-dataset-yolo-format).

## Why YOLOv11 (and which size)

- **YOLOv11** is the latest Ultralytics YOLO release: same easy API as v8, but faster and
  more accurate per parameter, with better small-object detection — useful for animals
  partially occluded by vegetation or shot at a distance.
- **`yolo11s.pt` (small)** is the recommended base checkpoint here: the dataset has only
  ~1,500 images across 4 classes, so nano/small models train quickly and avoid
  overfitting, while still giving strong accuracy. If you have a GPU with more headroom
  and want higher accuracy, try `yolo11m.pt`.

## Project files

| File | Purpose |
|---|---|
| `train.py` | Downloads the Kaggle dataset, fixes the `data.yaml` paths, fine-tunes YOLOv11 on it, evaluates, and saves `best.pt` |
| `app.py` | Streamlit app: upload an image or video and see detections, confidence scores, and a per-species count chart |
| `requirements.txt` | All Python dependencies |

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Kaggle credentials** (needed by `kagglehub` to download the dataset)
   - Get your API token from https://www.kaggle.com/settings → "Create New Token"
   - This downloads `kaggle.json`. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or
     `C:\Users\<you>\.kaggle\kaggle.json` (Windows).
   - Alternatively, set environment variables:
     ```bash
     export KAGGLE_USERNAME=your_username
     export KAGGLE_KEY=your_api_key
     ```

3. **Train the model**
   ```bash
   python train.py
   ```
   This will:
   - Download the dataset via `kagglehub`
   - Rebuild `data.yaml` with correct local paths
   - Train YOLOv11-small for 100 epochs (with early stopping, patience=20)
   - Save the best weights as `best.pt` in the project root
   - Run evaluation (mAP50, mAP50-95, precision, recall) on the test split

   > Training on a GPU is strongly recommended. On CPU only, reduce `epochs` and
   > `imgsz` in `train.py` to keep training time reasonable.

4. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   Open the URL Streamlit prints (usually `http://localhost:8501`).

## Using the app

- Choose **Image** or **Video** mode in the sidebar.
- Adjust **confidence** and **IoU** thresholds to tune sensitivity vs. false positives.
- Upload a file — detections are drawn as bounding boxes with class + confidence.
- A table and bar chart summarize detections per species.
- For video, you can download the annotated result.

If `best.pt` isn't found yet (i.e., you haven't run `train.py`), the app automatically
falls back to a generic pretrained `yolo11n.pt` so it still runs — but for real wildlife
class predictions (buffalo/elephant/rhino/zebra), you need to train first.

## Notes / tips

- If the dataset's folder names differ slightly (`val` vs `valid`), `train.py` already
  checks for both.
- To push higher accuracy: increase `epochs`, try `yolo11m.pt`, or add augmentation
  (Ultralytics enables sensible augmentations by default via `model.train(...)`).
- GPU training example is unchanged — Ultralytics auto-detects CUDA if available.
