"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import *

# def create_belleII_pipeline(**kwargs) -> Pipeline:
#     return pipeline(
#         [
#             node(
#                     func=tree_data_to_generator,
#                     inputs="decay_tree_events",
#                     outputs="generator_input",
#                     name="tree_data_to_generator"
#             ),
#             node(
#                     func=tree_data_to_discriminator,
#                     inputs="decay_tree_structure",
#                     outputs="discriminator_input",
#                     name="tree_data_to_discriminator"
#             ),
#             # node(
#             #         func=normalize_event,
#             #         inputs="generator_input",
#             #         outputs="generator_input_normalized",
#             #         name="normalize_generator"
#             # ),
#             node(
#                     func=conv_structure_to_lca_and_names,
#                     inputs=["artificial_decay", "decay_tree_structure"],
#                     outputs={
#                         "all_lca":"all_lca",
#                         "all_names":"all_names"
#                     },
#                     name="conv_structure_to_lca_and_names"
#             ),
#             node(
#                 func=lca_and_leaves_sort_into_modes,
#                 inputs=["artificial_decay", "all_lca", "all_names", "decay_tree_events"],
#                 outputs={
#                     "all_lca_mode_sorted":"all_lca_mode_sorted",
#                     "all_leaves_mode_sorted":"all_leaves_mode_sorted"
#                 },
#                 name="lca_and_leaves_sort_into_modes"
#             ),
#             node(
#                     func=shuffle_lca_and_leaves_in_mode,
#                     inputs=["artificial_decay", "all_lca_mode_sorted", "all_leaves_mode_sorted"],
#                     outputs={
#                         "all_lca_shuffled":"all_lca_shuffled",
#                         "all_leaves_shuffled":"all_leaves_shuffled"
#                     },
#                     name="shuffle_lca_and_leaves_in_mode"
#             ),
#         ]
#     )


def create_artificial_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=conv_structure_to_lca_and_names,
                inputs={
                    "pad_to": "params:pad_to",
                    "decay_tree_structure": "decay_tree_structure",
                },
                outputs={"all_lca": "all_lca", "all_names": "all_names"},
                name="conv_structure_to_lca_and_names",
            ),
            node(
                func=lca_and_leaves_sort_into_modes,
                inputs={
                    "n_topologies": "params:n_topologies",
                    "modes_names": "params:modes_names",
                    "train_events_per_top": "params:train_events_per_top",
                    "val_events_per_top": "params:val_events_per_top",
                    "test_events_per_top": "params:test_events_per_top",
                    "all_lca": "all_lca",
                    "all_names": "all_names",
                    "decay_tree_events": "decay_tree_events",
                },
                outputs={
                    "all_lca_mode_sorted": "all_lca_mode_sorted",
                    "all_leaves_mode_sorted": "all_leaves_mode_sorted",
                },
                name="lca_and_leaves_sort_into_modes",
            ),
            node(
                func=shuffle_lca_and_leaves_in_mode,
                inputs={
                    "modes_names": "params:modes_names",
                    "all_lca_mode_sorted": "all_lca_mode_sorted",
                    "all_leaves_mode_sorted": "all_leaves_mode_sorted",
                },
                outputs={
                    "all_lca_shuffled": "all_lca_shuffled",
                    "all_leaves_shuffled": "all_leaves_shuffled",
                },
                name="shuffle_lca_and_leaves_in_mode",
            ),
            node(
                func=lca_and_leaves_to_tuple_dataset,
                inputs={
                    "all_lca_shuffled": "all_lca_shuffled",
                    "all_leaves_shuffled": "all_leaves_shuffled",
                },
                outputs={"dataset_lca_and_leaves": "dataset_lca_and_leaves"},
                name="lca_and_leaves_to_tuple_dataset",
            ),
            # node(
            #         func=tuple_dataset_to_torch_tensor_dataset,
            #         inputs={"dataset_lca_and_leaves":"dataset_lca_and_leaves"},
            #         outputs={
            #             "torch_dataset_lca_and_leaves":"torch_dataset_lca_and_leaves",
            #         },
            #         name="tuple_dataset_to_torch_tensor_dataset"
            # )
        ]
    )


# def create_artificial_pipeline_no_shuffle(**kwargs) -> Pipeline:
#     return pipeline([
#         node(
#                 func=conv_structure_to_lca_and_names,
#                 inputs=["artificial_decay", "decay_tree_structure"],
#                 outputs={
#                     "all_lca":"all_lca",
#                     "all_names":"all_names"
#                 },
#                 name="conv_structure_to_lca_and_names"
#         ),
#         node(
#                 func=lca_and_leaves_sort_into_modes,
#                 inputs=["artificial_decay", "all_lca", "all_names", "decay_tree_events"],
#                 outputs={
#                     "all_lca_mode_sorted":"all_lca_mode_sorted",
#                     "all_leaves_mode_sorted":"all_leaves_mode_sorted"
#                 },
#                 name="lca_and_leaves_sort_into_modes"
#         ),
#         node(
#                 func=shuffle_lca_and_leaves_in_mode,
#                 inputs=["artificial_decay", "all_lca_mode_sorted", "all_leaves_mode_sorted"],
#                 outputs={
#                     "all_lca_shuffled":"all_lca_shuffled",
#                     "all_leaves_shuffled":"all_leaves_shuffled"
#                 },
#                 name="shuffle_lca_and_leaves_in_mode"
#         ),
#         node(
#                 func=lca_and_leaves_to_tuple_dataset,
#                 inputs=["all_lca_shuffled", "all_leaves_shuffled"],
#                 outputs={
#                     "dataset_lca_and_leaves":"dataset_lca_and_leaves"
#                 },
#                 name="lca_and_leaves_to_tuple_dataset"
#         ),
#         node(
#                 func=tuple_dataset_to_torch_tensor_dataset,
#                 inputs=["dataset_lca_and_leaves"],
#                 outputs={
#                     "torch_dataset_lca_and_leaves":"torch_dataset_lca_and_leaves",
#                 },
#                 name="tuple_dataset_to_torch_tensor_dataset"
#         )
#     ])
