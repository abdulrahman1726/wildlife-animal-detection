"""
app.py
------
Streamlit web app for the Wildlife Animal Detection project (YOLOv11).

Run with:
    streamlit run app.py

The app lets a user:
  - Upload an image or video (or use their webcam snapshot)
  - Run YOLOv11 inference using locally trained weights (best.pt)
  - View detected animals (buffalo, elephant, rhino, zebra) with bounding boxes
  - Adjust confidence / IoU thresholds
  - See a summary table of detections
"""

import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Wildlife Animal Detection - YOLOv11",
    page_icon="🦁",
    layout="wide",
)

MODEL_PATH_DEFAULT = "best.pt"      # produced by train.py
FALLBACK_MODEL = "yolo11n.pt"       # generic pretrained model if best.pt is missing
CLASS_NAMES = ["buffalo", "elephant", "rhino", "zebra"]


# --------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return YOLO(model_path)


def get_model():
    if os.path.exists(MODEL_PATH_DEFAULT):
        model_path = MODEL_PATH_DEFAULT
        st.sidebar.success(f"Loaded custom-trained weights: {model_path}")
    else:
        model_path = FALLBACK_MODEL
        st.sidebar.warning(
            "Custom weights 'best.pt' not found. Falling back to a generic "
            "pretrained YOLOv11 model (run train.py first for real wildlife results)."
        )
    return load_model(model_path)


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
iou_thres = st.sidebar.slider("IoU threshold", 0.0, 1.0, 0.45, 0.05)
input_mode = st.sidebar.radio("Input type", ["Image", "Video"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Classes detected:**")
st.sidebar.markdown(", ".join(CLASS_NAMES))

model = get_model()

# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------
st.title("🦁 Wildlife Animal Detection")
st.caption("YOLOv11 model trained on the Kaggle wildlife dataset (buffalo, elephant, rhino, zebra)")

if input_mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        with st.spinner("Running detection..."):
            start = time.time()
            results = model.predict(
                img_array, conf=conf_thres, iou=iou_thres, verbose=False
            )
            elapsed = time.time() - start

        result = results[0]
        annotated = result.plot()  # BGR numpy array with boxes drawn
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Detections")
            st.image(annotated_rgb, use_container_width=True)

        st.success(f"Inference completed in {elapsed:.2f}s")

        # Build a detections table
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            rows = []
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                rows.append(
                    {
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "x1": round(xyxy[0], 1),
                        "y1": round(xyxy[1], 1),
                        "x2": round(xyxy[2], 1),
                        "y2": round(xyxy[3], 1),
                    }
                )
            df = pd.DataFrame(rows)
            st.subheader("Detection details")
            st.dataframe(df, use_container_width=True)

            st.subheader("Counts per species")
            st.bar_chart(df["class"].value_counts())
        else:
            st.info("No animals detected. Try lowering the confidence threshold.")

else:  # Video mode
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)
        run_button = st.button("Run detection on video")

        if run_button:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path = video_path.replace(".mp4", "_detected.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frame_idx = 0

            status_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=conf_thres, iou=iou_thres, verbose=False)
                annotated_frame = results[0].plot()
                writer.write(annotated_frame)

                frame_idx += 1
                progress.progress(min(frame_idx / frame_count, 1.0))
                status_placeholder.text(f"Processing frame {frame_idx}/{frame_count}")

            cap.release()
            writer.release()

            st.success("Video processing complete!")
            st.video(out_path)

            with open(out_path, "rb") as f:
                st.download_button(
                    "Download annotated video",
                    data=f,
                    file_name="wildlife_detected.mp4",
                    mime="video/mp4",
                )

st.markdown("---")
st.caption(
    "Model: YOLOv11 (Ultralytics) | Dataset: ankanghosh651/object-detection-wildlife-dataset-yolo-format (Kaggle)"
)
