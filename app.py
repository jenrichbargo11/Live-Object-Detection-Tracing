import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import time
import os
from collections import defaultdict

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Live Object Detection & Tracing",
    page_icon="🎥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main app background */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #0f766e, #164e63);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

/* Animated background */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Main title */
h1 {
    text-align: center;
    font-size: 3rem !important;
    font-weight: 700;
    color: #f8fafc !important; 
    text-shadow: 
        0 0 8px rgba(125, 211, 252, 0.5),
        0 0 18px rgba(34, 211, 238, 0.3);
    letter-spacing: 1px;
}

/* Subtitle */
h3 {
    text-align: center;
    color: #67e8f9 !important;
    letter-spacing: 2px;
}

/* Paragraph text */
p {
    color: #e2e8f0 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.85);
    border-right: 2px solid #38bdf8;
    box-shadow: 0 0 20px rgba(56,189,248,0.4);
}

/* Sidebar header */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #38bdf8 !important;
}

/* Slider */
.stSlider > div > div {
    color: #38bdf8 !important;
}

/* Checkbox */
.stCheckbox label {
    color: white !important;
    font-weight: bold;
}

/* Video container */
[data-testid="stVideo"] {
    border: 4px solid #38bdf8;
    border-radius: 20px;
    box-shadow: 0 0 25px rgba(56,189,248,0.5);
    overflow: hidden;
}

/* Footer cards */
.footer-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 15px;
    border-radius: 16px;
    border: 1px solid rgba(56,189,248,0.4);
    margin: 8px 0;
    box-shadow: 0 0 15px rgba(56,189,248,0.2);
    transition: 0.3s;
}

.footer-card:hover {
    transform: scale(1.03);
    box-shadow: 0 0 25px rgba(56,189,248,0.5);
}

/* Horizontal line */
hr {
    border: 1px solid rgba(56,189,248,0.4);
}

</style>
""", unsafe_allow_html=True)

st.title("🎥 Live Object Detection & Tracing")
st.markdown("### AI-Powered Real-Time Object Detection using YOLOv8")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

save_frames = st.sidebar.checkbox("Save Detected Frames")

# Sidebar feature cards
st.sidebar.markdown("---")
st.sidebar.subheader("✨ Features")

features = [
    "✅ Detects objects",
    "✅ Tracks objects with IDs",
    "✅ Shows per-class counts",
    "✅ Saves snapshots"
]

for feature in features:
    st.sidebar.markdown(f"""
    <div class="footer-card">
        {feature}
    </div>
    """, unsafe_allow_html=True)

if save_frames:
    os.makedirs("captures", exist_ok=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- VIDEO PROCESSOR ----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.track_history = {}  # store ID -> label

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            results = model.track(
                img,
                conf=confidence,
                persist=True
            )

            annotated = img.copy()

            if results and results[0].boxes is not None:
                boxes = results[0].boxes

                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Confidence
                    conf_score = float(box.conf[0])

                    # Class label
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    # Track ID
                    track_id = int(box.id[0]) if box.id is not None else -1

                    # Save tracking history
                    if track_id != -1:
                        self.track_history[track_id] = label

                    # Draw rectangle
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Label with ID
                    text = f"ID {track_id} | {label} {conf_score:.2f}"
                    cv2.putText(
                        annotated,
                        text,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            # ---------------- PER-CLASS COUNT ----------------
            class_counts = defaultdict(int)

            for lbl in self.track_history.values():
                class_counts[lbl] += 1

            # Display counts
            y_offset = 40
            for cls, count in class_counts.items():
                text = f"{cls}: {count}"
                cv2.putText(
                    annotated,
                    text,
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )
                y_offset += 30

            # ---------------- SAVE FRAMES ----------------
            self.frame_count += 1
            if save_frames and self.frame_count % 60 == 0:
                filename = f"captures/frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated)

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- WEBRTC STREAM ----------------
ctx = webrtc_streamer(
    key="live-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "facingMode": "user"
        },
        "audio": False
    },
    async_processing=True
)

if ctx.state.playing:
    st.success("📷 Camera connected successfully!")
else:
    st.warning("⚠️ Click START and allow browser camera access.")

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    padding:18px;
    border-radius:16px;
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(56,189,248,0.35);
    box-shadow: 0 0 18px rgba(56,189,248,0.18);
    margin-top:20px;
">

<h4 style="color:#67e8f9; margin-bottom:10px;">
⚡ YOLOv8 Real-Time Vision System
</h4>

<p style="font-size:15px; color:#e2e8f0; margin:0;">
Live AI-powered object detection • Smart tracking • Frame analysis
</p>

<p style="font-size:13px; color:#94a3b8; margin-top:8px;">
Powered by <b>Ultralytics YOLOv8</b> • Streamlit WebRTC • OpenCV
</p>

<p style="font-size:12px; color:#64748b; margin-top:12px;">
🎥 Detect • Track • Analyze • Capture
</p>

</div>
""", unsafe_allow_html=True)