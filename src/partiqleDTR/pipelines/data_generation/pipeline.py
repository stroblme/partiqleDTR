"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_artificial_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=gen_structure_from_parameters,
                inputs={
                    "masses": "params:masses",
                    "fsp_masses": "params:fsp_masses",
                    "n_topologies": "params:n_topologies",
                    "max_depth": "params:max_depth",
                    "max_children": "params:max_children",
                    "min_children": "params:min_children",
                    "isp_weight": "params:isp_weight",
                    "iso_retries": "params:iso_retries",
                    "seed": "params:seed",
                },
                outputs={"decay_tree_structure": "decay_tree_structure"},
                name="gen_structure_from_parameters",
            ),
            node(
                func=gen_events_from_structure,
                inputs={
                    "n_topologies": "params:n_topologies",
                    "modes_names": "params:modes_names",
                    "train_events_per_top": "params:train_events_per_top",
                    "val_events_per_top": "params:val_events_per_top",
                    "test_events_per_top": "params:test_events_per_top",
                    "generate_unknown": "params:generate_unknown",
                    "seed": "params:seed",
                    "decay_tree_structure": "decay_tree_structure",
                },
                outputs={
                    "decay_tree_events": "decay_tree_events",
                    "decay_events_seeds": "decay_events_seeds",
                },
                name="gen_events_from_structure",
            ),
        ]
    )


# def create_belleII_pipeline(**kwargs) -> Pipeline:
#     return pipeline([
#         node(
#                 func=gen_decay_from_file,
#                 inputs="omega_decay",
#                 outputs={
#                     "decay_tree_structure":"decay_tree_structure",
#                     "decay_tree_events":"decay_tree_events"
#                 },
#                 name="gen_decay_from_file",
#             )
#     ])
