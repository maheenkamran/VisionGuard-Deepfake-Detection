import os, hashlib, sys
import pandas as pd
from PIL import Image

ROOT = "data"   

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
for cls in sorted(os.listdir(ROOT)):
    cdir = os.path.join(ROOT, cls)
    if not os.path.isdir(cdir):
        continue
    for dirpath, _, files in os.walk(cdir):
        for fn in files:
            p = os.path.join(dirpath, fn)
            row = dict(path=p, cls=cls, size_bytes=os.path.getsize(p),
                       sha256=sha256(p), width=None, height=None,
                       mode=None, format=None, corrupt=False)
            try:
                with Image.open(p) as im:
                    im.verify()              
                with Image.open(p) as im:    
                    im.load()                
                    row.update(width=im.width, height=im.height,
                               mode=im.mode, format=im.format)
            except Exception as e:
                row["corrupt"] = True
            rows.append(row)

df = pd.DataFrame(rows)
df["duplicate"] = df.duplicated("sha256", keep="first")   

os.makedirs("reports", exist_ok=True)
df.to_csv("reports/audit.csv", index=False)

ok = df[~df.corrupt]
print("Counts per class:\n", df.groupby("cls").size(), "\n")
print("Corrupt files:", df.corrupt.sum())
print("Exact duplicates:", df.duplicate.sum())
print(ok.groupby("cls")[["width", "height", "size_bytes"]]
        .agg(["min", "median", "max"]).T)
print(ok.groupby(["cls", "mode"]).size())
print(ok.groupby(["cls", "format"]).size())