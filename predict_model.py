import tensorflow as tf
import numpy as np

# Load model as inference layer
model = tf.keras.layers.TFSMLayer(
    "model.savedmodel",
    call_endpoint="serving_default"
)

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip().split(' ')[1] for line in f.readlines()]

print("Model loaded successfully!")
print("Labels:", labels)