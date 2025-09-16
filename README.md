# 📦 google-mediapipe-gst-python-implementation

<p align="center">
  <img width="45%" style="margin: 10px; alt="Screenshot 2025-07-02 at 7 30 54 PM" src="https://github.com/user-attachments/assets/ec9f64d4-4c68-41ed-aaa2-18fdf31fb0c0" />
  <img width="45%" style="margin: 10px; alt="Screenshot 2025-07-02 at 7 30 54 PM" src="https://github.com/user-attachments/assets/3e2c62b3-e2ef-4850-9b13-7282221d20ab" /><br>
</p>

### Demo:
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/dc63492a-ce12-479b-81af-b61e4d9a737d" alt="GIF Demo" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/f7b0ec45-dfcd-44e9-bc77-9bb159b24277" alt="App Screenshot" width="400"/></td>
  </tr>
</table>


## 🚀 Real-Time GStreamer Video Processing with Shared Memory and Batch Processing

**google-mediapipe-gst-python-implementation** is a high-performance, real-time video processing pipeline that:

- Streams frames from a GStreamer pipeline
- Batches frames efficiently into shared memory
- Distributes batches across multiple processes using multiprocessing queues
- Supports scalable, parallel feature extraction (e.g. using Mediapipe)

This project is designed for **low-latency, high-throughput computer vision pipelines.**

## Architecture

<img width="1284" height="709" alt="diagram-export-9-16-2025-6_21_37-PM" src="https://github.com/user-attachments/assets/47075317-0690-4f1f-9db9-d5f87b15ff67" />

---

## 📂 Project Structure

```plaintext
.
├── gaze_tracker.py       # gaze tracking function
├── utils.py              # Shared utilities (e.g., batch handling, frame reshaping)
├── requirements.txt      # Python package dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Features

- ✅ Real-time GStreamer frame ingestion
- ✅ Batch processing for improved memory and CPU efficiency
- ✅ Shared memory transport (zero-copy frame passing)
- ✅ Multi-process consumer architecture (supports parallel Mediapipe tasks)
- ✅ Easily configurable batch size and frame resolution
- ✅ Scalable for multi-camera or high-frame-rate pipelines

---

## 🛠️ Requirements

- Python 3.8+
- GStreamer 1.18+

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
numpy
opencv-python
matplotlib
```

---

## 🚀 How It Works

1. **Producer:**
   - Reads frames from the GStreamer pipeline using appsink
   - Writes frames into a shared memory batch buffer
   - Signals when batches are ready via a lightweight queue

2. **Consumers:**
   - Listen for batch-ready signals
   - Read batches directly from shared memory (zero-copy)
   - Process each frame independently (e.g. using Mediapipe feature extraction)

---

## 📦 Usage

Run the pipeline:

```bash
python manage.py runserver
```

**Customize:**
- Adjust `BATCH_SIZE` and `FRAME_SHAPE` in `utils.py`
- Integrate your own Mediapipe feature extraction functions in `consumer.py`

---

## 📸 Example Pipeline

```python
pipeline = Gst.parse_launch('videotestsrc ! video/x-raw,format=BGR,width=1280,height=720 ! appsink name=sink')
```

You can easily replace this with your own GStreamer input (webcam, video file, live stream, etc.)

---

## 🔗 Future Work

- Support for double-buffering to enable full parallel read/write cycles
- Multi-camera ingestion
- GPU-accelerated processing support
- Optional Kafka or distributed queue integration

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributions

Pull requests and improvements are welcome! Please open an issue first to discuss major changes.

---

## ✨ Credits

Built by http://github.com/KarthikMgk for scalable, real-time vision pipelines.
