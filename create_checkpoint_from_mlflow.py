import os
import yaml
import pickle
import mlflow
import datetime

#ecc74ac0312340b48d157667da192a45
id = str(input("Enter experiment id: \t"))
run = mlflow.get_run(id)
start_time = datetime.datetime.utcfromtimestamp(run.info.start_time/1000).strftime('%Y-%m-%d %H:%M:%S')
input(f"Found a run from {start_time} at git hash {run.data.tags['git_hash'][:6]}. Continue?")
artifact_uri = run.info.artifact_uri

with open(os.path.join(artifact_uri[7:], "model.yml"), "r") as f:
    model_state_dict = yaml.unsafe_load(f)

with open(os.path.join(artifact_uri[7:], "optimizer.yml"), "r") as f:
    optimizer_state_dict = yaml.unsafe_load(f)

start_epoch = 1

checkpoint = {
                "start_epoch": start_epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
            }

print(f"Checkpoint opened, writing to file at {os.path.join(artifact_uri[7:], 'checkpoint.pickle')}")

with open(os.path.join(artifact_uri[7:], "checkpoint.pickle"), "wb") as f:
    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done. You'll need to overwrite the checkpoint.pickle file in /data/08_reporting in order to use it.")