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
  - Run kedro visualization: ```poetry run kedro viz```
  - Open in [browser](http://127.0.0.1:4141/)
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
  - Run mlflow dashboard: ```poetry run mlflow ui```
  - Open in [browser](http://127.0.0.1:5000)


## :pray: Acknowledgement

See the [Helmholtz-AI-Energy/BaumBauen](https://github.com/Helmholtz-AI-Energy/BaumBauen) project for a fully classical implementation.
Part of the code from mentioned repository is reused within this project.