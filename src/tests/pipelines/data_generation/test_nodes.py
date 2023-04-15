from partiqleDTR.pipelines.data_generation.nodes import gen_structure_from_parameters, gen_events_from_structure
from partiqleDTR.pipelines.data_processing.nodes import _conv_decay_to_lca

class TestDataConsistency:
    def test_decay_structure(self, data_generation_parameters):
        structures_a = gen_structure_from_parameters(
            **data_generation_parameters
        )["decay_tree_structure"]

        structures_b = gen_structure_from_parameters(
            **data_generation_parameters
        )["decay_tree_structure"]

        for structure_a, structure_b in zip(structures_a, structures_b): 
            lca_a, name = _conv_decay_to_lca(structure_a)
            lca_b, name = _conv_decay_to_lca(structure_b)

            assert (lca_a == lca_b).all()

    def test_decay_data(self, data_generation_parameters, data_processing_parameters):

        structures = gen_structure_from_parameters(
            **data_generation_parameters
        )["decay_tree_structure"]

        events_a = gen_structure_from_parameters(
            **data_generation_parameters
        )["decay_tree_events"]

        events_a = gen_structure_from_parameters(
            **data_generation_parameters
        )["decay_tree_events"]