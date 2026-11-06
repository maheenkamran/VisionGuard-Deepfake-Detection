import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.inference import Classifier

c = Classifier("models/champion.keras", 0.31)
for path in sys.argv[1:]:
    print(path, "\n", c.predict(open(path, "rb").read()))