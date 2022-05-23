"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import gen_decay_from_file, gen_structure_from_parameters, gen_events_from_structure

def create_artificial_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=gen_structure_from_parameters,
                inputs="artificial_decay",
                outputs={
                    "decay_tree_structure":"decay_tree_structure"
                },
                name="gen_structure_from_parameters",
        ),
        node(
                func=gen_events_from_structure,
                inputs=["artificial_decay", "decay_tree_structure"],
                outputs={
                    "decay_tree_events":"decay_tree_events"
                },
                name="gen_events_from_structure",
        )
    ])

def create_belleII_pipeline(**kwargs) -> Pipeline:
    return pipeline([
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
