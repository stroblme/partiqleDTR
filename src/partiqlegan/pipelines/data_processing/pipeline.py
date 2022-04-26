"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import tree_data_to_generator, tree_data_to_discriminator, normalize

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=tree_data_to_generator,
                inputs="decay_tree_events",
                outputs="generator_input",
                name="tree_data_to_generator"
        ),
        node(
                func=tree_data_to_discriminator,
                inputs="decay_tree_structure",
                outputs="discriminator_input",
                name="tree_data_to_discriminator"
        ),
        node(
                func=normalize,
                inputs="generator_input",
                outputs="generator_input_normalized",
                name="normalize_generator"
        ),
        node(
                func=normalize,
                inputs="discriminator_input",
                outputs="discriminator_input_normalized",
                name="normalize_discriminator"
        )
    ])
