import io
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class Classifier:
    def __init__(self, path, threshold):
        self.model = tf.keras.models.load_model(path)    
        self.threshold = threshold

    def predict(self, data: bytes):
        img = Image.open(io.BytesIO(data)).convert("RGB")
        x = tf.image.resize(np.asarray(img, dtype="float32"), (224, 224))  # same as training
        x = preprocess_input(x.numpy())[None] # [-1, 1] scaling
        p = float(self.model.predict(x, verbose=0)[0][0]) # P(real)
        label = "Real" if p >= self.threshold else "AI-generated"
        conf = p if label == "Real" else 1 - p
        return label, conf