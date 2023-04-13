"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.17.7
"""
import numpy as np
from typing import Dict, Tuple, Any, List
from phasespace import GenParticle, nbody_decay
from phasespace.fromdecay import GenMultiDecay
from decaylanguage import DecFileParser, DecayChainViewer
import tensorflow as tf
import pandas as pd

from ..data_processing.nodes import _conv_decay_to_lca


def gen_decay_from_file(decaylanguage: Dict[str, Any]) -> Dict[Dict, Tuple[List, List]]:
    MOTHER_PARTICLE = (
        decaylanguage["MOTHER_PARTICLE"] if "MOTHER_PARTICLE" in decaylanguage else None
    )
    STABLE_PARTICLES = (
        decaylanguage["STABLE_PARTICLES"] if "STABLE_PARTICLES" in decaylanguage else ()
    )
    DECAY_FILE = decaylanguage["DECAY_FILE"] if "DECAY_FILE" in decaylanguage else None
    N_EVENTS = decaylanguage["N_EVENTS"] if "N_EVENTS" in decaylanguage else None
    VIEW_GRAPH = decaylanguage["VIEW_GRAPH"] if "VIEW_GRAPH" in decaylanguage else None

    parser = DecFileParser(DECAY_FILE)
    parser.parse()

    decay_chain = parser.build_decay_chains(
        MOTHER_PARTICLE, stable_particles=STABLE_PARTICLES
    )

    dcv = DecayChainViewer(decay_chain)
    dcv.graph.render(filename="decayGraph", format="pdf", view=VIEW_GRAPH, cleanup=True)

    decay_process = GenMultiDecay.from_dict(decay_chain)

    weights, events = decay_process.generate(n_events=N_EVENTS)

    return {"decay_tree_structure": decay_chain, "decay_tree_events": (weights, events)}


# def gen_nbody_decay_data(
#     parameters: Dict[str, np.ndarray]
# ) -> Dict[str, np.ndarray]:

#     particles = dict()
#     N_EVENTS = parameters["N_EVENTS"] if "N_EVENTS" in parameters else None


#     # Retrive particle masses and number of events
#     for key, value in parameters.items():
#         if "MASS" in key:
#             particleName = key.replace("_MASS","")
#             particles[f"{particleName}_0"] = GenParticle(particleName, value)

#     # Add some extra particles
#     particles["Pp_1"] = GenParticle("Pp_1", parameters["Pp_MASS"])
#     particles["Pm_1"] = GenParticle("Pm_1", parameters["Pm_MASS"])
#     particles["P0_1"] = GenParticle("P0_1", parameters["P0_MASS"])

#     # Build the decay tree
#     particles["D0_0"].set_children(particles["Kp_0"], particles["Pm_0"], particles["P0_0"])
#     particles["O_0"].set_children(particles["Pp_1"], particles["Pm_1"], particles["P0_1"])
#     particles["Bp_0"].set_children(particles["D0_0"], particles["O_0"], particles["Pp_0"])

#     # Generate a few events
#     weights, events = particles["Bp_0"].generate(n_events=N_EVENTS)

#     for i, p in enumerate(events):
#         events[p] = np.array(events[p]).reshape(4,N_EVENTS)

#     return {
#         "decay_tree_structure": decay_chain,
#         "decay_tree_events": (weights, events)
#     }


def gen_structure_from_parameters(
    masses: List[int],
    fsp_masses: List[int],
    n_topologies: int,
    max_depth: int,
    max_children: int,
    min_children: int,
    isp_weight: int,
    iso_retries: int,
    seed: int,
) -> Dict[str, np.ndarray]:

    # particles = dict()

    # masses = parameters["MASSES"] if "MASSES" in parameters else None
    # fsp_masses = parameters["FSP_MASSES"] if "FSP_MASSES" in parameters else None
    # n_topologies = parameters["N_TOPOLOGIES"] if "N_TOPOLOGIES" in parameters else None
    # max_depth = parameters["MAX_DEPTH"] if "MAX_DEPTH" in parameters else None
    # max_children = parameters["MAX_CHILDREN"] if "MAX_CHILDREN" in parameters else None
    # min_children = parameters["MIN_CHILDREN"] if "MIN_CHILDREN" in parameters else None
    # isp_weight = parameters["ISP_WEIGHT"] if "ISP_WEIGHT" in parameters else None
    # TRAIN_EVENTS_PER_TOP = parameters["TRAIN_EVENTS_PER_TOP"] if "TRAIN_EVENTS_PER_TOP" in parameters else None
    # VAL_EVENTS_PER_TOP = parameters["VAL_EVENTS_PER_TOP"] if "VAL_EVENTS_PER_TOP" in parameters else None
    # TEST_EVENTS_PER_TOP = parameters["TEST_EVENTS_PER_TOP"] if "TEST_EVENTS_PER_TOP" in parameters else None
    # # SEED = parameters["SEED"] if "SEED" in parameters else None
    # iso_retries = parameters["ISO_RETRIES"] if "ISO_RETRIES" in parameters else None
    # generate_unknown = parameters["GENERATE_UNKNOWN"] if "GENERATE_UNKNOWN" in parameters else None

    """Generate a PhaseSpace dataset

    Args:
        root (str or Path): root folder
        masses (list): intermediate particle masses, root mass is masses[0]
        fsp_masses (list): list of final state particles
        topologies (int): number of decay tree topologies to generate, twice this number
                          for validation and thrice this number for testing
        max_depth (int): maximum allowed depth of the decay trees
        max_children (int): maximum allowed number of children for intermediate particles
        min_children (int): minumum required number of children for intermediate particles
                            (can fall short if kinematically impossible)
        isp_weight (float): relative weight of intermediate state particle probability (higher is more likely than fsp)
        train_events_per_top(int): number of training samples generated per decay tree
        val_events_per_top(int): number of validation samples generated per decay tree
        test_events_per_top(int): number of test samples generated per decay tree
        seed(int): RNG seed
        iso_retries(int):  if this is <= 0, does not perform isomorphism checks between generated topologies.
                           if > 0 gives the number of retries to ensure non-isomorphic topologies before raising
    """

    # import tensorflow as tf

    # if SEED is not None:
    # np.random.seed(SEED)
    # This is supposed to be supported as a global seed for Phasespace but doesn't work
    # Instead we set the seed below in the calls to generate()
    # tf.random.set_seed(np.random.randint(np.iinfo(np.int32).max))

    if int(max_depth) <= 1:
        raise ValueError("Tree needs to have at least two levels")

    if int(min_children) < 2:
        raise ValueError("min_children must be two or more")

    masses = sorted(masses, reverse=True)
    fsp_masses = sorted(fsp_masses, reverse=True)
    if not set(masses).isdisjoint(set(fsp_masses)):
        raise ValueError(
            "Particles are only identified by their masses. Final state particle masses can not occur in intermediate particle masses."
        )

    topology_isomorphism_invariates = []

    total_topologies = n_topologies
    # if generate_unknown:
    #     total_topologies = 3 * n_topologies

    decay_tree_structure = list()

    rd = np.random.default_rng(seed)  # rd.choice #TODO: check that
    # n_seeds = total_topologies * max(
    #     1, iso_retries
    # )  # TODO: this is chosen on gut feeling..

    for i in range(total_topologies):
        # t_rd = np.random.default_rng(rd.integers(1000*total_topologies))
        # NOTE generate tree for a topology
        for j in range(max(1, iso_retries)):
            queue = []
            root_node = GenParticle("root", masses[0])
            queue.append((root_node, 1))
            name = 1
            next_level = 1
            # l_rd = np.random.default_rng(t_rd.integers(1000*max_children))
            while len(queue) > 0:
                node, level = queue.pop(0)
                if next_level <= level:
                    next_level = level + 1
                num_children = rd.integers(min_children, max_children + 1)

                total_child_mass = 0
                children = []

                # Mass we have to play with
                avail_mass = node._mass_val
                # Add an insurance to make sure it's actually possible to generate two children
                if avail_mass <= (2 * min(fsp_masses)):
                    raise ValueError(
                        "Any ISP mass given has to be larger than two times the smallest FSP mass."
                    )

                # c_rd = np.random.default_rng(rd.integers(num_children))
                for k in range(num_children):
                    # Only want to select children from mass/energy available
                    avail_mass -= total_child_mass

                    # Check we have enough mass left to generate another child
                    if avail_mass <= min(fsp_masses):
                        break

                    # use fsps if last generation or at random determined by number of possible masses and isp weight
                    if (
                        next_level == max_depth
                        or avail_mass <= min(masses)
                        or rd.random()
                        < (1.0 * len(fsp_masses))
                        / ((1.0 * len(fsp_masses)) + (isp_weight * len(masses)))
                    ):
                        child_mass = rd.choice(
                            [n for n in fsp_masses if (n < avail_mass)]
                        )
                    else:
                        child_mass = rd.choice(
                            [n for n in masses if (n < avail_mass)]
                        )
                    total_child_mass += child_mass

                    if total_child_mass > node._mass_val:
                        break

                    child = GenParticle(str(name), child_mass)
                    children.append(child)
                    name += 1
                    if child_mass in masses:
                        queue.append((child, next_level))

                node.set_children(*children)

            # NOTE if iso_retries given, check if topology already represented in dataset
            top_iso_invar = assign_parenthetical_weight_tuples(root_node)
            if iso_retries <= 0 or top_iso_invar not in topology_isomorphism_invariates:
                topology_isomorphism_invariates.append(top_iso_invar)
                break
            if j == (iso_retries - 1):
                raise RuntimeError(
                    "Could not find sufficient number of non-isomorphic topologies."
                )
                # print("Could not find sufficient number of non-isomorphic topologies.")
                # continue

        decay_tree_structure.append(root_node)

    # event_data = gen_train_data(parameters, decay_tree_structure)
    # convert_to_lca(parameters, decay_tree_structure, event_data["decay_tree_events"])

    return {"decay_tree_structure": decay_tree_structure}


def assign_parenthetical_weight_tuples(node):
    """
    Args:
        node(phasespace.GenParticle

    After A. Aho, J. Hopcroft, and J. Ullman The Design and Analysis of Computer Algorithms. Addison-Wesley Publishing Co., Reading, MA, 1974, pp. 84-85.
    Credits to Alexander Smal

    """
    if not node.has_children:
        return f"({node.get_mass()})"

    child_tuples = [assign_parenthetical_weight_tuples(c) for c in node.children]
    child_tuples.sort()
    child_tuples = "".join(child_tuples)

    return f"({node.get_mass()}{child_tuples})"


def gen_events_from_structure(
    n_topologies: int,
    modes_names: List[str],
    train_events_per_top: int,
    val_events_per_top: int,
    test_events_per_top: int,
    seed: int,
    decay_tree_structure: Dict[str, np.ndarray],
):
    # n_topologies = parameters["N_TOPOLOGIES"] if "N_TOPOLOGIES" in parameters else None
    # modes_names = parameters["MODES_NAMES"] if "MODES_NAMES" in parameters else None
    # train_events_per_top = parameters["TRAIN_EVENTS_PER_TOP"] if "TRAIN_EVENTS_PER_TOP" in parameters else None
    # val_events_per_top = parameters["VAL_EVENTS_PER_TOP"] if "VAL_EVENTS_PER_TOP" in parameters else None
    # test_events_per_top = parameters["TEST_EVENTS_PER_TOP"] if "TEST_EVENTS_PER_TOP" in parameters else None
    # generate_unknown = parameters["GENERATE_UNKNOWN"] if "GENERATE_UNKNOWN" in parameters else None
    # seed = parameters["SEEDS"] if "SEEDS" in parameters else None

    events_per_top = {
        modes_names[0]: train_events_per_top,
        modes_names[1]: val_events_per_top,
        modes_names[2]: test_events_per_top,
    }

    all_weights = {mode: list() for mode in modes_names}
    all_events = {mode: list() for mode in modes_names}

    rd = np.random.default_rng(seed)  # rd.choice #TODO: check that
    # n_seeds = len(decay_tree_structure) * len(modes_names)
    # seeds = [rd.integers(n_seeds * 1000) for i in range(n_seeds)]

    actual_seeds = []
    # iterate topologies
    for i, root_node in enumerate(decay_tree_structure):
        # NOTE generate leaves and labels for training, validation, and testing
        modes = []
        # For topologies not in the training set, save them to a different subdir
        # save_dir = Path(root, 'unknown')

        # uncommented the following lines at 220926 since there is no obvious reason why we need e.g. 10, 20, 30 samples in train, val, test respectively.
        # ----------------------------------------------------------------
        # if i < n_topologies or not generate_unknown:
        #     modes = modes_names
        #     # save_dir = Path(root, 'known')
        # elif i < (2 * n_topologies):
        #     modes = modes_names[1:]
        # else:
        #     modes = modes_names[2:]
        # save_dir.mkdir(parents=True, exist_ok=True)
        # ----------------------------------------------------------------
        # modes = [m for m, e in events_per_top.items() if e >= i]
        modes = modes_names

        if i >= n_topologies:
            break

        for mode in modes:
            num_events = events_per_top[mode]
            l_seed = rd.integers(1000*len(decay_tree_structure))
            tf_rng = tf.random.Generator.from_seed(l_seed)
            # random.seed(l_seed)
            # np.random.seed(l_seed)
            # tf.random.set_seed(l_seed)
            weights, events = root_node.generate(num_events, seed=tf_rng)
            actual_seeds.append(l_seed)

            all_weights[mode].append(weights)
            all_events[mode].append(events)

    return {
        "decay_tree_events": (all_weights, all_events),
        "decay_events_seeds": actual_seeds,
    }

def evaluate_seeds_fsps(
    masses: List[int],
    fsp_masses: List[int],
    n_topologies: int,
    max_depth: int,
    max_children: int,
    min_children: int,
    isp_weight: int,
    iso_retries: int,
    seed: int,
    n_tries: int,
):
    seeds_fsp_dict = {}

    for i in range(n_tries):
        cur_seed = seed + i

        result = gen_structure_from_parameters(
                masses,
                fsp_masses,
                n_topologies,
                max_depth,
                max_children,
                min_children,
                isp_weight,
                iso_retries,
                cur_seed,
        )

        structures = result["decay_tree_structure"]

        all_fsps = []
        for structure in structures: 
            lca, name = _conv_decay_to_lca(structure)
            all_fsps.append(lca.shape[0])

        max_fsps = max(all_fsps)

        print(f"| {cur_seed} | {max_fsps} |")

        if max_fsps not in seeds_fsp_dict:
            seeds_fsp_dict[max_fsps] = []
        seeds_fsp_dict[max_fsps].append(cur_seed)

    return {"seeds_fsp_dict":pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in seeds_fsp_dict.items() ])).fillna(0.0).astype(int)} #from https://stackoverflow.com/questions/19736080/creating-dataframe-from-a-dictionary-where-entries-have-different-lengths
    