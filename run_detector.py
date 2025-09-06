import cv2
import time
import os
import numpy as np

from utils.sound_player import SoundPlayer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CFG_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
NAMES_PATH = os.path.join(MODELS_DIR, "coco.names")

# Load labels
if not os.path.exists(NAMES_PATH):
    raise FileNotFoundError(f"Missing {NAMES_PATH}. Run download_models.py first.")
with open(NAMES_PATH, "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

if "dog" not in LABELS:
    raise RuntimeError("'dog' class not found in coco.names")

DOG_CLASS_ID = LABELS.index("dog")

# Build net
if not os.path.exists(CFG_PATH) or not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError("YOLO model files missing. Run download_models.py to fetch them.")

net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up video capture
cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    raise RuntimeError("Cannot access webcam. Plug in a camera or change the index in VideoCapture.")

# Sound system
player = SoundPlayer(BASE_DIR)
last_play_time = 0
cooldown = 2.0  # seconds between sound plays

CONF_THRESH = 0.3
NMS_THRESH = 0.3
INPUT_SIZE = 416

def decide_level(count):
    if count == 0:
        return None
    if count == 1:
        return "low"
    if 2 <= count <= 3:
        return "mid"
    return "high"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    (H, W) = frame.shape[:2]

    # Create blob and run inference
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > CONF_THRESH:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)

    dog_count = 0
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, w, h) = boxes[i]
            label = LABELS[class_ids[i]]
            conf = confidences[i]
            if class_ids[i] == DOG_CLASS_ID:
                dog_count += 1
                # Draw dog boxes thicker
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(frame, f"dog {conf:.2f}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # UI overlay
    level = decide_level(dog_count)
    status = "SAFE" if dog_count == 0 else f"DOGS: {dog_count} → LEVEL: {level.upper()}"
    color = (0, 200, 0) if dog_count == 0 else ((0, 200, 200) if level == "low" else ((0, 165, 255) if level == "mid" else (0, 0, 255)))
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    # Flash screen edges when active (simulate street light flash)
    if level:
        thickness = 25
        cv2.rectangle(frame, (0,0), (W-1,H-1), color, thickness)

    # Play sound with cooldown
    now = time.time()
    if level and (now - last_play_time) > cooldown:
        player.play(level)
        last_play_time = now

    cv2.imshow("Smart Dog Pole - AI Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()