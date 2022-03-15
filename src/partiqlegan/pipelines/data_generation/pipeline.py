"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import gen_nbody_decay_data, gen_decay_from_file

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=gen_nbody_decay_data,
                inputs="phasespace",
                outputs="nbody_decay_data",
                name="gen_nbody_decay_data",
            ),
        node(
                func=gen_decay_from_file,
                inputs="decaylanguage",
                outputs="decay_tree_data",
                name="gen_decay_from_file",
            )
    ])
