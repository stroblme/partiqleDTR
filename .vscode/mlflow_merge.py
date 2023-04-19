import glob
import os
import yaml
import shutil

mlflow_path = "./mlruns/1"

merge_from = ["/storage/mstrobl/PartiqleGAN", "/home/mstrobl/ssd_storage/partiqleDTR", "/local/scratch/mstrobl/partiqleDTR"]
replace_with = "/home/lc3267/Documents/CodeWorkspace/PartiqleGAN"

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

        for path in merge_from:
            if path in artifact_uri:
                content["artifact_uri"] = artifact_uri.replace(path, replace_with)
                mark_for_update = True

    if mark_for_update:
        with open(os.path.join(r, "meta.yaml"), "w") as f:
            yaml.safe_dump(content, f)