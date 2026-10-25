# exploratory data analysis
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

df = pd.read_csv("reports/audit.csv")

#sample images grid for each class
fig, axes = plt.subplots(2, 8, figsize=(20, 6)) #2 rows, 8 cols 

for row, cls in enumerate(["AiArtData", "RealArt"]): #both classes
    paths = df[df.cls == cls].path.sample(8, random_state=1) # 8 samples
    for col, p in enumerate(paths):
        axes[row, col].imshow(Image.open(p).convert("RGB"))
        axes[row, col].axis("off")

fig.suptitle("Top row: AI-generated, Bottom row: real images", fontsize=16)
plt.tight_layout()
plt.savefig("figures/sample_grid.png", dpi=150)
plt.show()

#making histogram
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

bins = np.logspace(4, 7.1, 40)   

for cls in ["AiArtData", "RealArt"]:
    d = df[df.cls == cls]
    axes[0].hist(d.width, bins=40, alpha=0.5, label=cls)
    axes[1].hist(d.height, bins=40, alpha=0.5, label=cls)
    axes[2].hist(df[df.cls == cls].size_bytes, bins=bins, alpha=0.5, label=cls)
    axes[2].set_xscale("log")

axes[0].set_title("Width (pixels)")
axes[1].set_title("Height (pixels)")
axes[2].set_title("File size (bytes)")
axes[2].set_xscale("log")

for ax in axes:
    ax.set_ylabel("Number of images")
    ax.legend()

plt.tight_layout()
plt.savefig("figures/histograms.png", dpi=150)
plt.show()