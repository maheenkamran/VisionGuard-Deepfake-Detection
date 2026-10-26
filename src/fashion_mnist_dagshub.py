import mlflow
import mlflow.keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from dotenv import load_dotenv
load_dotenv()

(x, y), _ = keras.datasets.mnist.load_data()
x = x[:10000] / 255.0
y = y[:10000]

mlflow.set_experiment("toy_mnist_dagshub")

for lr in [0.01, 0.0001]:
    model = keras.Sequential([
        layers.Input((28, 28)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    with mlflow.start_run(run_name=f"lr_{lr}"):
        history = model.fit(x, y, epochs=5, validation_split=0.2, verbose=0)
        acc = history.history["val_accuracy"][-1]

        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", 5)
        mlflow.log_metric("val_accuracy", acc)

        plt.plot(history.history["val_accuracy"])
        plt.title(f"val acc, lr={lr}")
        plt.savefig("curve.png")
        plt.close()
        mlflow.log_artifact("curve.png")

        mlflow.keras.log_model(model, name="model")

        print("lr", lr, "val accuracy", acc)