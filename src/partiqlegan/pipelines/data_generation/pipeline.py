"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import gen_decay_from_file, gen_nbody_decay_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=gen_nbody_decay_data,
                inputs="simple_decay",
                outputs="nbody_decay_data",
                name="gen_nbody_decay_data",
            ),
        node(
                func=gen_decay_from_file,
                inputs="omega_decay",
                outputs={
                    "decay_tree_structure":"decay_tree_structure",
                    "decay_tree_events":"decay_tree_events"
                },
                name="gen_decay_from_file",
            )
    ])
