import mlflow
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# id="2d3aff6fc96c49f28c74dc9f057fecee"
id = str(input("Enter experiment id: \t"))
run = mlflow.get_run(id)
start_time = datetime.datetime.utcfromtimestamp(run.info.start_time / 1000).strftime(
    "%Y-%m-%d %H:%M:%S"
)
input(
    f"Found a run from {start_time} at git hash {run.data.tags['git_hash'][:6]}. Continue?"
)

client = mlflow.tracking.MlflowClient()

metrics = [
    "num_selected_params",
]

metric_histories = dict()
for metric in metrics:
    metric_history = client.get_metric_history(id, metric)
    metric_histories[metric] = ([], [])
    for metric_step in metric_history:
        metric_histories[metric][0].append(metric_step.step)
        metric_histories[metric][1].append(metric_step.value)


fig = plt.figure(figsize=(12, 7))
ax = plt.gca()

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams.update({"font.size": 22})
plt.rcParams.update({"axes.titlesize": 22})

cmap = mpl.cm.get_cmap("tab10")

hline_offset = 0.5

handles = []
labels = []

for i, metric in enumerate(metrics):
    if "logic" in metric:
        continue
    if "loss" in metric:
        best = min(metric_histories[metric][1])
        best_idx = np.argmin(metric_histories[metric][1])
    elif "perfect" in metric or "accuracy" in metric:
        best = max(metric_histories[metric][1])
        best_idx = np.argmax(metric_histories[metric][1])
    elif "num" in metric:
        best = None

    ax.plot(*metric_histories[metric], color=cmap(i), label=metric)
    ax.set_xticks(
        metric_histories[metric][0][1 :: (len(metric_histories[metric][0]) // 10)]
    )  # we start counting at 1, so it looks better if we have the max epoch shown
    ax.tick_params(labelsize=SMALL_SIZE)

    if best is not None:
        # highlight the best line
        plt.axhline(y=best, xmin=hline_offset/(len(metric_histories[metric][0])), xmax=(best_idx+hline_offset)/(len(metric_histories[metric][0])), color=cmap(i), linewidth=0.8, linestyle=':', label=f"best_{metric}")

        handles.append(cmap(i))
        labels.append(f"best_{metric}")

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
# ax.set_facecolor("#FAFAFA")
ax.set_facecolor("#FFFFFF")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Epoch", fontsize=MEDIUM_SIZE)

if len(metrics) == 1:
    ax.set_ylabel("Metric", fontsize=MEDIUM_SIZE)
else:
    ax.set_ylabel(f"{metrics[0].replace('_', ' ').upper()}", fontsize=MEDIUM_SIZE)
ax.set_ylim([0.0, 0.9])

plt.axvline(x=1, color="#666666", linewidth=0.7)
plt.axhline(y=0, color="#666666", linewidth=0.7)
ax.tick_params(axis="both", which="both", length=0)
# fig.patch.set_facecolor("#FAFAFA")
fig.patch.set_facecolor("#FFFFFF")
# plt.grid(color="#F1F1F1")
ax.grid(False)
# ax.xaxis.grid(True)
# Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), facecolor="#FAFAFA", framealpha=0.3)
if len(metrics) > 1:
    legend = plt.legend(
        bbox_to_anchor=(1.04, 1),
        loc="lower left",
        ncol=len(labels) // 2,
        framealpha=0.3,
        frameon=False,
    )


    def export_legend(legend, filename):
        leg_fig = legend.figure
        leg_fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(leg_fig.dpi_scale_trans.inverted())
        leg_fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    export_legend(legend, "legend.pdf")

    ax.get_legend().remove()

# no title
# title = input("Gimme a catchy title: \t")
# plt.title(title, fontsize=BIGGER_SIZE)
plt.savefig("output.pdf", bbox_inches="tight")
