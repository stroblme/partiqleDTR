import glob
import os
import yaml
import shutil

mlflow_path = "./mlruns/1"

runs = glob.glob(os.path.join(mlflow_path, "*"))

for r in runs:
    mark_for_update = False
    if not os.path.isdir(r):
        continue

    with open(os.path.join(r, "meta.yaml"), "r") as f:
        try:
            content = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
        

        artifact_uri = content["artifact_uri"]

        if "/home/mstrobl/ssd_storage/partiqleDTR" in artifact_uri:
            content["artifact_uri"] = artifact_uri.replace("/home/mstrobl/ssd_storage/partiqleDTR", "/home/lc3267/Documents/CodeWorkspace/PartiqleGAN")
            mark_for_update = True

    if mark_for_update:
        with open(os.path.join(r, "meta.yaml"), "w") as f:
            yaml.safe_dump(content, f)
