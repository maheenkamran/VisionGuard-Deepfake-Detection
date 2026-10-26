import sys
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from mlflow import MlflowClient
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

load_dotenv()

NAME = "visionguard"
ALIAS = "champion"
THRESHOLD = 0.31   # F1-optimal threshold 

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/load_champion.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    client = MlflowClient()

    mv = client.get_model_version_by_alias(NAME, ALIAS)
    local_model_path = client.download_artifacts(
        run_id=mv.run_id, 
        path="keras_file/r5_ftlr_1e-4.keras"
    )

    model = tf.keras.models.load_model(local_model_path)

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    
    x = preprocess_input(np.array(img, dtype="float32"))[None]

    p = float(model.predict(x, verbose=0)[0][0])

    result = "REAL" if p >= THRESHOLD else "AI-generated"
    print(f"{result} | p(real)={p:.3f}")

if __name__ == "__main__":
    main()