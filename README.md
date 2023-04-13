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

### Choosing a `proper` seed

The following table shows some seeds that you can use to generate a dataset with a specific number of FSPs:

|7|5|4|6|8|9|3|
|---|---|---|---|---|---|---|
|1110|1112|1113|1114|1146|1262|1633|
|1111|1117|1118|1116|1150|1547|1708|
|1115|1119|1122|1126|1171|1672|0|
|1120|1121|1127|1128|1175|1685|0|
|1138|1123|1130|1131|1199|1926|0|
|1145|1124|1152|1135|1216|2017|0|
|1147|1125|1161|1136|1281|2035|0|

Note that entries with $0$ mean, that there were no seeds found within the range of tested seeds between $1110$ and $2110$.

The seed is also being used for generating decay events, so it might be usefull to evaluate different seeds despite they result in the same number of FSPs.
That's why multiple seeds are provided in the table above, as e.g. seed $1110$ and $1111$ will **not** yield the same dataset.


## :pray: Acknowledgement

See the [Helmholtz-AI-Energy/BaumBauen](https://github.com/Helmholtz-AI-Energy/BaumBauen) project for a fully classical implementation.
Part of the code from mentioned repository is reused within this project.