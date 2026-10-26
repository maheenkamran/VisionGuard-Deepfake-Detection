import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline import make_ds

ds = make_ds("splits/train.csv", True, batch=9, preprocess=None, img=224, augment=True)
x, y = next(iter(ds))

fig, ax = plt.subplots(3, 3, figsize=(7, 7))
for i, a in enumerate(ax.ravel()):
    a.imshow(x[i].numpy().astype("uint8"))
    a.set_title("Real" if y[i] == 1 else "AI")
    a.axis("off")
fig.suptitle("One augmented training batch")
fig.savefig("reports/augmented_batch.png", dpi=120, bbox_inches="tight")
print("saved reports/augmented_batch.png")