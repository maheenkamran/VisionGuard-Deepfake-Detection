import os
import numpy as np
import pandas as pd
import tensorflow as tf

IMG = 224
CLASSES = sorted(["AiArtData", "RealArt"])      # RealArt = 1, AiArtData = 0
AUTOTUNE = tf.data.AUTOTUNE
DATA_ROOT = os.getenv("DATA_ROOT", "data")      


def make_ds(csv, training, batch=32, preprocess=None, img=IMG,
            augment=True, max_samples=None, seed=42):
    df = pd.read_csv(csv)
    if max_samples:                              
        df = df.sample(n=min(max_samples, len(df)), random_state=seed).reset_index(drop=True)

    paths = (DATA_ROOT + "/" + df["rel_path"]).values
    labels = (df["cls"] == CLASSES[1]).astype("float32").values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(p, y):
        x = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        x = tf.image.resize(x, (img, img))       
        x.set_shape((img, img, 3))
        return x, y

    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()                              

    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
        if augment:                             
            aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ])
            ds = ds.map(lambda x, y: (aug(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)

    if preprocess is not None:                  
        ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)

    return ds.batch(batch).prefetch(AUTOTUNE)


def get_labels(ds):
    return np.concatenate([y.numpy() for _, y in ds]).astype(int)