"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import gen_nbody_decay_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=gen_nbody_decay_data,
                inputs="phasespace",
                outputs="nbody_decay_data",
                name="gen_nbody_decay_data",
            )
    ])
