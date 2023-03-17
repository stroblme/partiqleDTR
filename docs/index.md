# Documentation

Attempt to solve the particle decay tree reconstruction problem using hybrid quantum machine learning architectures.

## Setup

Assuming [poetry](https://python-poetry.org/) is installed, simply run the VSCode task "Setup" or manually:

```bash
.vscode/setup.sh
```

If you use VSCode, you can run mlflow and kedro-viz as tasks.
There are also launch configurations for the kedro experiments.


## Project layout

This project uses [kedro](https://kedro.org) and [mlflow](https://mlflow.org) as framework and for experiment tracking purposes.
Below are some of the main files and folders:

    .vscode/setup.sh        # Starting point on a freshly cloned repo
    .vscode/setup_dev.sh    # Includes development modules and runs default build steps
    src/partiqlegan         # Kedro pipelines and nodes
    src/tests               # Stuff for testing
    data                    # Kedro data folders
