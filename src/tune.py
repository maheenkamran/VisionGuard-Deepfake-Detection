import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import EarlyStopping

from pipeline import make_ds
from models import compile_model

pre = keras.applications.mobilenet_v2.preprocess_input
train_ds = make_ds("splits/train.csv", True, 32, pre, 224)
val_ds = make_ds("splits/val.csv", False, 32, pre, 224)


def build(hp):
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                          input_shape=(224, 224, 3))
    n = hp.Choice("unfrozen_layers", [0, 20, 40, 60])
    base.trainable = n > 0
    for i, l in enumerate(base.layers):
        l.trainable = (n > 0) and i >= len(base.layers) - n \
                      and not isinstance(l, L.BatchNormalization)
    inp = keras.Input((224, 224, 3))
    x = base(inp, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(hp.Float("dropout", 0.1, 0.6, step=0.1))(x)
    out = L.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    compile_model(model, hp.Choice("lr", [1e-2, 1e-3, 1e-4, 1e-5]))
    return model


tuner = kt.RandomSearch(build, objective=kt.Objective("val_auc", "max"),
                        max_trials=10, directory="kt_vg",
                        project_name="visionguard", overwrite=True, seed=42)

tuner.search(train_ds, validation_data=val_ds, epochs=6,
             callbacks=[EarlyStopping("val_loss", patience=2, restore_best_weights=True)])

tuner.results_summary()
print("BEST:", tuner.get_best_hyperparameters(1)[0].values)

print("\n| Rank | unfrozen_layers | dropout | lr | val_auc |\n|---|---|---|---|---|")
for i, t in enumerate(tuner.oracle.get_best_trials(10), 1):
    h = t.hyperparameters
    print(f"| {i} | {h.get('unfrozen_layers')} | {h.get('dropout'):.1f} | {h.get('lr')} | {t.score:.4f} |")