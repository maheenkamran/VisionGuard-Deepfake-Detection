import os
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("reports/audit.csv")
clean = df[(df.corrupt == False) & (df.duplicate == False)]

# 70% train, 30% leftover
train, tmp = train_test_split(clean, test_size=0.30, stratify=clean.cls, random_state=42)
# 30% leftover is split in half, 15% val, 15% test
val, test = train_test_split(tmp, test_size=0.50, stratify=tmp.cls, random_state=42)

os.makedirs("splits", exist_ok=True)
# rel_path (not path) so the CSVs work on any machine, including Kaggle
train[["rel_path", "cls"]].to_csv("splits/train.csv", index=False)
val[["rel_path", "cls"]].to_csv("splits/val.csv", index=False)
test[["rel_path", "cls"]].to_csv("splits/test.csv", index=False)

print("train", len(train), train.cls.value_counts().to_dict())
print("val", len(val), val.cls.value_counts().to_dict())
print("test", len(test), test.cls.value_counts().to_dict())