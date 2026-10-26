import os
from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient

load_dotenv()

NAME = "visionguard"
RUN_V1 = "3de39466208c4d5d8d28c7d22c1d7338"    # t2_mobilenet
RUN_V2 = "052d6e7fd8d9469b823a6f17412c9d35"    # r5_ftlr_1e-4

client = MlflowClient()

try:
    client.create_registered_model(NAME)
except Exception:
    pass

v1_uri = f"{mlflow.get_run(RUN_V1).info.artifact_uri}/keras_file/t2_mobilenet.keras"
v2_uri = f"{mlflow.get_run(RUN_V2).info.artifact_uri}/keras_file/r5_ftlr_1e-4.keras"

v1 = client.create_model_version(name=NAME, source=v1_uri, run_id=RUN_V1)
v2 = client.create_model_version(name=NAME, source=v2_uri, run_id=RUN_V2)

client.set_registered_model_alias(NAME, "champion", v2.version)

print("versions:", v1.version, v2.version, "champion ->", v2.version)