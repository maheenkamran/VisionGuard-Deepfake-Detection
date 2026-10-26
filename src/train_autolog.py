import argparse
import os
import re
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay)
import mlflow
import mlflow.keras

from pipeline import make_ds, get_labels
from models import (build_baseline, build_transfer, unfreeze_last_block,
                    compile_model, get_preprocess, trainable_names)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="mobilenetv2",
                   choices=["cnn", "mobilenetv2", "efficientnetb0", "resnet50"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ft_lr", type=float, default=1e-5)
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--aug", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--ft_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default="t2_mobilenet_autolog")
    p.add_argument("--experiment", default="visionguard_main")
    p.add_argument("--splits", default="splits")
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def dvc_hash():
    for path in ("data.dvc", "data/raw.dvc"):
        try:
            return re.search(r"md5:\s*(\S+)", open(path).read()).group(1)
        except Exception:
            continue
    return "unknown"


class Stage2Metrics(keras.callbacks.Callback):
    """Metrics only (no params), used for stage 2 because autolog is off there."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def callbacks(ckpt, patience, extra=None):
    cbs = [
        keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]
    return cbs + (extra or [])


def main():
    a = parse()
    load_dotenv()
    keras.utils.set_random_seed(a.seed)
    os.environ["PYTHONHASHSEED"] = str(a.seed)
    os.makedirs("models", exist_ok=True)
    out_dir = f"reports/{a.run_name}"
    os.makedirs(out_dir, exist_ok=True)

    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("https://dagshub.com"):
        raise SystemExit("MLFLOW_TRACKING_URI is not set to DagsHub.")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(a.experiment)

    mlflow.keras.autolog(log_models=False, log_datasets=False)

    patience = 8 if a.backbone == "cnn" else 4
    pre = get_preprocess(a.backbone)
    train_ds = make_ds(f"{a.splits}/train.csv", True, a.batch, pre, a.img,
                       bool(a.aug), a.max_samples, a.seed)
    val_ds = make_ds(f"{a.splits}/val.csv", False, a.batch, pre, a.img,
                     max_samples=a.max_samples, seed=a.seed)
    test_ds = make_ds(f"{a.splits}/test.csv", False, a.batch, pre, a.img,
                      max_samples=a.max_samples, seed=a.seed)

    t0 = time.time()
    with mlflow.start_run(run_name=a.run_name):
        skip = {"epochs", "batch"}
        mlflow.log_params({k: v for k, v in vars(a).items() if k not in skip})
        mlflow.set_tags({"git_commit": git_commit(), "dvc_data_version": dvc_hash(),
                         "seed": str(a.seed), "tf_version": tf.__version__,
                         "gpu": str(bool(tf.config.list_physical_devices("GPU"))),
                         "logging_mode": "autolog"})

        histories, trainable_log = [], {}
        if a.backbone == "cnn":
            model, base = build_baseline(a.img, a.dropout, a.lr), None
        else:
            model, base = build_transfer(a.backbone, a.img, a.dropout, a.lr)
        trainable_log["stage1"] = trainable_names(model)
        print("Stage 1 trainable layers:", trainable_log["stage1"])

        stage1_ckpt = f"models/{a.run_name}_stage1.keras"
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs,
                       callbacks=callbacks(stage1_ckpt, patience))
        histories.append(h1.history)
        best1 = min(h1.history["val_loss"])
        final_ckpt = stage1_ckpt
        n1 = len(h1.history["loss"])

        if base is not None and a.ft_epochs > 0:
            mlflow.keras.autolog(disable=True)
            unfreeze_last_block(base, a.backbone, a.ft_lr)
            compile_model(model, a.ft_lr)
            trainable_log["stage2"] = trainable_names(model)
            print("Stage 2 trainable layers:", trainable_log["stage2"])
            stage2_ckpt = f"models/{a.run_name}_stage2.keras"
            h2 = model.fit(train_ds, validation_data=val_ds,
                           initial_epoch=n1, epochs=n1 + a.ft_epochs,
                           callbacks=callbacks(stage2_ckpt, patience, [Stage2Metrics()]))
            histories.append(h2.history)
            if min(h2.history["val_loss"]) < best1:
                final_ckpt = stage2_ckpt

        model = keras.models.load_model(final_ckpt)

        cat = {k: sum([h[k] for h in histories], []) for k in histories[0]}
        best_epoch = int(np.argmin(cat["val_loss"])) + 1
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for axi, k in zip(ax, ["loss", "accuracy"]):
            axi.plot(range(1, len(cat[k]) + 1), cat[k], label="train")
            axi.plot(range(1, len(cat[k]) + 1), cat["val_" + k], label="val")
            axi.axvline(best_epoch, color="r", ls="--", label=f"best epoch {best_epoch}")
            if len(histories) > 1:
                axi.axvline(n1 + 0.5, color="gray", ls=":", label="fine-tune start")
            axi.set_xlabel("epoch"); axi.set_title(k); axi.legend()
        fig.savefig(f"{out_dir}/curves.png", dpi=120, bbox_inches="tight")

        pv = model.predict(val_ds, verbose=0).ravel()
        yv = get_labels(val_ds)
        ths = np.linspace(0.05, 0.95, 91)
        thr = float(ths[np.argmax([f1_score(yv, pv >= t, zero_division=0) for t in ths])])

        pt = model.predict(test_ds, verbose=0).ravel()
        yt = get_labels(test_ds)
        pred = (pt >= thr).astype(int)
        metrics = {
            "test_accuracy": accuracy_score(yt, pred),
            "test_precision": precision_score(yt, pred, zero_division=0),
            "test_recall": recall_score(yt, pred, zero_division=0),
            "test_f1": f1_score(yt, pred, zero_division=0),
            "test_roc_auc": float(roc_auc_score(yt, pt)),
            "val_auc_best": float(roc_auc_score(yv, pv)),
            "threshold": thr, "best_epoch": best_epoch,
            "n_params": model.count_params(), "train_seconds": time.time() - t0,
        }
        print(metrics)

        cm_fig, cm_ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix(yt, pred),
                               display_labels=["AI", "Real"]).plot(ax=cm_ax)
        cm_ax.set_title(f"Confusion matrix (thr={thr:.2f})")
        cm_fig.savefig(f"{out_dir}/confusion_matrix.png", dpi=120, bbox_inches="tight")
        roc_fig, roc_ax = plt.subplots()
        RocCurveDisplay.from_predictions(yt, pt, ax=roc_ax)
        roc_fig.savefig(f"{out_dir}/roc_curve.png", dpi=120, bbox_inches="tight")

        np.save(f"{out_dir}/test_probs.npy", pt)
        model.save(f"models/{a.run_name}.keras")

        mlflow.log_metrics(metrics)
        mlflow.log_dict(trainable_log, "trainable_layers.json")
        mlflow.log_figure(fig, "curves.png")
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        mlflow.log_figure(roc_fig, "roc_curve.png")
        mlflow.log_artifacts(a.splits, "splits")
        mlflow.log_artifact(f"{out_dir}/test_probs.npy")
        subprocess.run("pip freeze > requirements_run.txt", shell=True)
        mlflow.log_artifact("requirements_run.txt")
        mlflow.log_artifact(f"models/{a.run_name}.keras", "keras_file")


if __name__ == "__main__":
    main()