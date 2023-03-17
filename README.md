# partiqlegan


decayfile -> decaytree -> decayevents

masses -> decaytopologies (containing decaytrees) -> decayevents


## Installation

1. Clone the repository
2. Create a venv
3. Install the requirements using
```bash
pip compile 
```
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



