import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

run = sys.argv[1]           
thr = float(sys.argv[2])     

df = pd.read_csv("splits/test.csv")
df["prob"] = np.load(f"reports/{run}/test_probs.npy")      
df["true"] = (df["cls"] == "RealArt").astype(int)
df["pred"] = (df["prob"] >= thr).astype(int)

audit = pd.read_csv("reports/audit.csv")[["rel_path", "width", "height", "size_bytes", "format"]]
df = df.merge(audit, on="rel_path", how="left")

err = df[df["true"] != df["pred"]].copy()
print(f"{len(err)} errors out of {len(df)} test images")
print("fake called real:", int(((err.true == 0) & (err.pred == 1)).sum()))
print("real called fake:", int(((err.true == 1) & (err.pred == 0)).sum()))

df["wrong"] = df["true"] != df["pred"]
print("\nMedian size by correctness:")
print(df.groupby("wrong")[["width", "height", "size_bytes"]].median())
print("\nError rate by format:")
print(df.groupby("format")["wrong"].agg(["mean", "count"]))

show = err.sample(n=min(10, len(err)))    
print("\nThe 10 images shown:")
print(show[["rel_path", "true", "pred", "prob", "width", "height", "size_bytes", "format"]]
      .to_string())

fig, ax = plt.subplots(2, 5, figsize=(16, 7))
for a, (_, r) in zip(ax.ravel(), show.iterrows()):
    a.imshow(Image.open("data/" + r.rel_path).convert("RGB"))
    a.set_title(f"true={'Real' if r.true else 'AI'}, pred={'Real' if r.pred else 'AI'}\n"
                f"p(real)={r.prob:.2f}  {int(r.width)}x{int(r.height)}", fontsize=9)
    a.axis("off")
fig.tight_layout()
fig.savefig(f"reports/{run}/errors.png", dpi=120, bbox_inches="tight")
print(f"\nsaved reports/{run}/errors.png")