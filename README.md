# PartiqleDTR

Repository of the Partiqle Decay Tree Reconstruction (DTR) project.
The PartiqleDTR project attempts to tackle the problem of particle decay tree reconstruction using hybrid quantum-classical machine learning approaches.

## :rocket:  Setup and Run

1. Clone the repository
2. Install dependencies
   - **Option A:** Poetry installed on your system
   ```
   poetry install
   ```
   - **Option B:** Create a venv, and install dependencies from ```pyproject.toml``` manually
3. Create a dataset and run training using the existing parameters:
   ```
   kedro run
   ```

## :wrench: Configuration

*The command prefix ```poetry run``` can be omitted if you're using an already activated venv*

- Pipeline Configuration:
  - Open visualization: ```poetry run kedro viz```
  1. Data Generation:
     - Builds decay tree
     - Generates decay events
  2. Data Processing
     - Preprocessing events
     - Generating datasets
  3. Data Science
     - Model creation
     - Training
- Parameters can be adjusted in the following locations:
  - ```conf/base/parameters/data_generation.yml```
  - ```conf/base/parameters/data_processing.yml```
  - ```conf/base/parameters/data_science.yml```
- Runs are being recorded using MLFlow
  - Open dashboard: ```poetry run mlflow ui```

## Notes

torch_scatter needs gcc-c++ and python3-devel packages to build successfully.

with poetry you need to release the tensorflow-probability dependency from phasespace such that the line in ```.venv/lib/python3.10/site-packages/phasespace-1.8.0.dist-info/METADATA``` becomes
```
Requires-Dist: tensorflow-probability (>=0.15)
```
To achieve this, you may want to run
```
poetry add phasespace
```
prior to the installation of the other packages using
```
poe

changed module dependencies in phasespace:

before:

Requires-Dist: tensorflow (<2.8,>=2.6)
Requires-Dist: tensorflow-probability (<0.14,>=0.11)
Requires-Dist: keras (<2.7)

after:

Requires-Dist: tensorflow (>=2.6)
Requires-Dist: tensorflow-probability (>=0.11)
Requires-Dist: keras (<2.10)

then 
pip install -U tensorflow should update keras, numpy etc
pip install zfit (uninstall first if already installed manually)


### MLflow

upgrade kedro
run kedro mlflow init



