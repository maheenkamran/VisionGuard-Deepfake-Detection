from tensorflow import keras
from tensorflow.keras import layers as L


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")])


def get_preprocess(backbone):
    if backbone == "mobilenetv2":
        return keras.applications.mobilenet_v2.preprocess_input    
    if backbone == "resnet50":
        return keras.applications.resnet50.preprocess_input        
    if backbone == "efficientnetb0":
        return keras.applications.efficientnet.preprocess_input    
    return None                                                     


def build_baseline(img=224, dropout=0.4, lr=1e-3):
    model = keras.Sequential([
        L.Input((img, img, 3)),
        L.Rescaling(1 / 255.0),
        L.Conv2D(32, 3, padding="same", activation="relu"),
        L.BatchNormalization(),
        L.MaxPooling2D(),
        L.Conv2D(64, 3, padding="same", activation="relu"),
        L.BatchNormalization(),
        L.MaxPooling2D(),
        L.Conv2D(128, 3, padding="same", activation="relu"),
        L.BatchNormalization(),
        L.MaxPooling2D(),
        L.GlobalAveragePooling2D(),
        L.Dropout(dropout),
        L.Dense(64, activation="relu"),
        L.Dense(1, activation="sigmoid"),
    ], name="baseline_cnn")
    compile_model(model, lr)
    return model


def _backbone(name, img):
    cls = {"mobilenetv2": keras.applications.MobileNetV2,
           "efficientnetb0": keras.applications.EfficientNetB0,
           "resnet50": keras.applications.ResNet50}[name]
    return cls(include_top=False, weights="imagenet", input_shape=(img, img, 3))


def build_transfer(backbone="mobilenetv2", img=224, dropout=0.3, lr=1e-3):
    base = _backbone(backbone, img)
    base.trainable = False                       # stage 1: frozen
    inp = keras.Input((img, img, 3))
    x = base(inp, training=False)                # BatchNorm stays in inference mode
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out, name=f"visionguard_{backbone}")
    compile_model(model, lr)
    return model, base


# name prefixes of the LAST block of each backbone
LAST_BLOCK = {
    "mobilenetv2": ("block_16", "Conv_1"),
    "efficientnetb0": ("block7", "top_"),
    "resnet50": ("conv5_block3",),
}


def unfreeze_last_block(base, backbone, lr=None):
    """Unfreeze only the last block. Caller must recompile afterwards."""
    prefixes = LAST_BLOCK[backbone]
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith(prefixes)
        if isinstance(layer, L.BatchNormalization):
            layer.trainable = False              # keep BN statistics stable


def trainable_names(model):
    """Names of trainable layers (looks inside the backbone). Needed for 2.3 evidence."""
    names = []
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            names += [l.name for l in layer.layers if l.trainable]
        elif layer.trainable and layer.weights:
            names.append(layer.name)
    return names