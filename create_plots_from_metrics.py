import mlflow
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# id="2d3aff6fc96c49f28c74dc9f057fecee"
id = str(input("Enter experiment id: \t"))
run = mlflow.get_run(id)
start_time = datetime.datetime.utcfromtimestamp(run.info.start_time/1000).strftime('%Y-%m-%d %H:%M:%S')
input(f"Found a run from {start_time} at git hash {run.data.tags['git_hash'][:6]}. Continue?")

client = mlflow.tracking.MlflowClient()

metrics = ["train_loss", "val_loss", "train_accuracy", "val_accuracy", "train_logic_accuracy", "val_logic_accuracy", "train_perfect_lcag", "val_perfect_lcag"]

metric_histories = dict()
for metric in metrics:
    metric_history = client.get_metric_history(id, metric)
    metric_histories[metric] = ([], [])
    for metric_step in metric_history:
        metric_histories[metric][0].append(metric_step.step)
        metric_histories[metric][1].append(metric_step.value)


fig = plt.figure(figsize=(12,6))
ax = plt.gca()

cmap = mpl.cm.get_cmap('Dark2_r')

for i, metric in enumerate(metrics):
    if "loss" in metric:
        best = min(metric_histories[metric][1])
        best_idx = np.argmin(metric_histories[metric][1])
    elif "perfect" or "accuracy" in metric:
        best = max(metric_histories[metric][1])
        best_idx = np.argmax(metric_histories[metric][1])

    ax.plot(*metric_histories[metric], color=cmap(i), label=metric)
    ax.set_xticks(metric_histories[metric][0][1::(len(metric_histories[metric][0])//10)]) # we start counting at 1, so it looks better if we have the max epoch shown
    plt.axhline(y=best, xmin=1/len(metric_histories[metric][0]), xmax=(1+best_idx)/(len(metric_histories[metric][0])+1), color=cmap(i), linewidth=0.7, linestyle=':', label=f"best_{metric}")

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
ax.set_facecolor("#FAFAFA")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.axvline(x=1, color="#666666", linewidth=0.7)
ax.tick_params(axis=u'both', which=u'both',length=0)
fig.patch.set_facecolor("#FAFAFA")
plt.grid(color="#F1F1F1")

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), facecolor="#FAFAFA", framealpha=1)
title = input("Gimme a catchy title: \t")
plt.title(title)
plt.savefig("test.png", bbox_inches='tight')


