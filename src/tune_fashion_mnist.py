import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import EarlyStopping

(xtr, ytr), _ = keras.datasets.fashion_mnist.load_data()

def build(hp):
    m = keras.Sequential([
        L.Input((28, 28)), L.Flatten(),
        L.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        L.Dense(10, activation="softmax")])
    m.compile(keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3, 1e-4])),
              "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=6,
                        directory="kt_toy", project_name="fashion",
                        overwrite=True, seed=42)

# Toy data: raw arrays + validation_split (no train_ds / val_ds here)
tuner.search(xtr / 255.0, ytr, epochs=5, validation_split=0.2,
             callbacks=[EarlyStopping("val_loss", patience=2, restore_best_weights=True)])

tuner.results_summary()
print("BEST:", tuner.get_best_hyperparameters(1)[0].values)

print("\n| Rank | units | lr | val_accuracy |\n|---|---|---|---|")
for i, t in enumerate(tuner.oracle.get_best_trials(6), 1):
    print(f"| {i} | {t.hyperparameters.get('units')} | {t.hyperparameters.get('lr')} | {t.score:.4f} |")