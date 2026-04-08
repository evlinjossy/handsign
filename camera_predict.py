import cv2
import tensorflow as tf
import numpy as np
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)

prev_label = ""
current_label = ""
count = 0
threshold = 10   # number of frames to confirm

# Load model
model = tf.keras.layers.TFSMLayer(
    "model.savedmodel",
    call_endpoint="serving_default"
)

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip().split(' ')[1] for line in f.readlines()]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)

    # Preprocess
    img = cv2.resize(roi, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model(img)
    prediction = list(prediction.values())[0].numpy()
    index = np.argmax(prediction)
    label = labels[index]

    # Stability check
    if label == current_label:
        count += 1
    else:
        current_label = label
        count = 0

    # Speak only when stable
    if count == threshold and label != prev_label:
        engine.say(label)
        engine.runAndWait()
        prev_label = label

    # Display
    cv2.putText(frame, f"Sign: {label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()