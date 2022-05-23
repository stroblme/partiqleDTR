"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import tree_data_to_generator, tree_data_to_discriminator, normalize, conv_structure_to_lca_and_names, shuffle_lca_and_names

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
                func=conv_structure_to_lca_and_names,
                inputs=["artificial_decay", "decay_tree_structure"],
                outputs={
                    "all_lca":"all_lca",
                    "all_names":"all_names"
                },
                name="conv_structure_to_lca_and_names"
        ),
        node(
                func=shuffle_lca_and_names,
                inputs=["artificial_decay", "all_lca", "all_names", "decay_tree_events"],
                outputs={
                    "all_lca_shuffled":"all_lca_shuffled",
                    "all_leave_shuffled":"all_leave_shuffled"
                },
                name="shuffle_lca_and_names"
        )
    ])
