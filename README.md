
# SmartDogPole – AI Dog-Detection Street Pole (Python + OpenCV)

This project detects **dogs** using your **webcam** and a **YOLO v4-tiny** model.
It **counts** dogs and plays different **sound levels** to safely scare them away.
It also simulates a **flashing street-light** border on the video.

> Grade-5 friendly demo + competition-ready AI idea.

---

## 🔧 What you need
- Python 3.9–3.11
- Webcam
- Speakers (laptop/desktop)
- Internet (one-time) to download model files

## 📦 Setup (Windows/Mac/Linux)

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

# 2) Go to project folder
cd SmartDogPole

# 3) Install dependencies
pip install -r requirements.txt

# 4) Download model files (YOLOv4-tiny cfg/weights + coco.names)
python download_models.py

# 5) Run!
python run_detector.py
```

Press **q** to quit the window.

## 🔊 Sound logic
- **0 dogs** → silent
- **1 dog** → plays `low.wav` (~4 kHz)
- **2–3 dogs** → plays `mid.wav` (~10 kHz)
- **4+ dogs** → plays `high.wav` (~15 kHz)

> Note: Most laptop speakers cannot play true ultrasonic (>20 kHz), so we use **higher-pitched audible tones** for the demo. Keep **volume moderate** for comfort.

## 🧠 How detection works
We use **OpenCV DNN** with **YOLOv4-tiny** trained on COCO dataset (which includes class `dog`).
The script draws red boxes on detected dogs and shows a status banner with current **level**.

## 🧪 Tips for a great demo
- Aim the webcam at a **printed photo** or **toy dog** or **YouTube dog video**.
- Ensure **good lighting**.
- Increase confidence by keeping the dog **larger in the frame**.

## 🛠️ Customize
Open `run_detector.py` and adjust:
- `CONF_THRESH` (default 0.3)
- `NMS_THRESH` (default 0.3)
- `cooldown` between sounds (default 2 seconds)

You can also swap model to **YOLOv3-tiny** if you prefer:
- Download `yolov3-tiny.cfg` and `yolov3-tiny.weights`
- Update the `CFG_PATH` and `WEIGHTS_PATH` in `run_detector.py`

## 🧯 Safety & Ethics
- This is a **non-harm** demo. Do **not** use dangerously loud volume.
- Use responsibly; prefer **animal-friendly** solutions and coordinate with local authorities for real deployments.

## 📁 Project structure
```
SmartDogPole/
  download_models.py
  requirements.txt
  run_detector.py
  README.md
  models/                # YOLO files go here (auto-downloaded)
  sounds/
    low.wav
    mid.wav
    high.wav
  utils/
    sound_player.py
```

## 🐞 Troubleshooting
**Q: Webcam not found?**  
Edit `cv2.VideoCapture(0)` → try `1` or `2`.

**Q: No sound?**  
`pygame` mixer may fail on some systems; Windows fallback uses `winsound.Beep`.

**Q: Detection is slow?**  
Make the window smaller; ensure only one camera app is open.

---

Happy building and stay safe! 🌆🐶
