# 🎥 Live Object Detection & Tracing using YOLOv8

## 📌 Project Overview

This project is a **real-time object detection and tracking web application** built with **Streamlit**, **YOLOv8**, **OpenCV**, and **WebRTC**.

It uses your computer’s webcam to:

* Detect objects in real time
* Track detected objects with unique IDs
* Count detected object classes
* Save annotated frames automatically
* Display live AI-powered analysis directly in the browser

The system is powered by **Ultralytics YOLOv8**, a state-of-the-art deep learning model for object detection.

---

## 🚀 Features

✅ Real-time webcam object detection
✅ Object tracking with unique IDs
✅ Per-class object counting
✅ Adjustable confidence threshold
✅ Save detected frames automatically
✅ Interactive browser camera access
✅ Modern animated cyber-style UI
✅ Built with Streamlit WebRTC

---

## 🛠 Technologies Used

* **Python**
* **Streamlit**
* **Streamlit-WebRTC**
* **Ultralytics YOLOv8**
* **OpenCV**
* **AV**
* **Collections (defaultdict)**

---

## 📂 Project Structure

```bash
live-object-detection/
│── app.py
│── requirements.txt
│── README.md
│── captures/
```

---

## ⚙️ Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/live-object-detection.git
cd live-object-detection
```

---

### 2. Install Dependencies

```bash
pip install streamlit
pip install streamlit-webrtc
pip install ultralytics
pip install opencv-python
pip install av
```

Or install all at once:

```bash
pip install streamlit streamlit-webrtc ultralytics opencv-python av
```

---

### 3. Run the Application

```bash
streamlit run app.py
```

---

## 📷 How to Use

### Step 1

Run the app using:

```bash
streamlit run app.py
```

---

### Step 2

Open your browser at:

```bash
http://localhost:8501
```

---

### Step 3

Click **START**

---

### Step 4

Allow browser camera permission

---

### Step 5

Start detecting and tracking objects live

---

## 🎛 Controls

### Confidence Threshold

Adjusts the minimum confidence level for object detection.

* Lower value → detects more objects
* Higher value → detects only highly confident detections

---

### Save Detected Frames

When enabled, the system automatically saves annotated frames every 60 frames.

Saved files are stored in:

```bash
captures/
```

---

## 🧠 How It Works

The app processes live webcam frames using YOLOv8.

### Detection Pipeline

1. Capture webcam frame
2. Convert frame into NumPy array
3. Run YOLOv8 object detection
4. Track detected objects
5. Draw bounding boxes
6. Display object labels + confidence scores
7. Count detected classes
8. Stream annotated video back to browser

---

## 📊 Output Display

The app displays:

* Bounding boxes
* Object labels
* Confidence score
* Tracking ID
* Per-class object count

Example:

```text
ID 4 | person 0.92
ID 7 | bottle 0.88
```

---

## 🔧 Troubleshooting

### Camera Not Opening

Make sure:

* Browser camera permission is enabled
* No other app is using the webcam
* You are running locally via:

```bash
streamlit run app.py
```

---

### Module Not Found Error

Install missing packages:

```bash
pip install package-name
```

---

### Webcam Permission Denied

In Chrome/Edge:

**Settings → Privacy → Camera → Allow**

---

## 📈 Future Improvements

Possible upgrades:

* Face detection
* Custom-trained YOLO models
* Object analytics dashboard
* Motion heatmaps
* Detection history logs
* Cloud deployment

---

## 👨‍💻 Developer
Author: Jenrich Bargo
It was Developed as an **AI-powered computer vision project** using:

* Ultralytics YOLOv8
* Streamlit WebRTC
* OpenCV

---

## 📜 License

This project is for educational and research purposes.

---

## 🎯 System Footer Motto

**Detect • Track • Analyze • Capture**
