"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.17.7
"""
import numpy as np
from typing import Dict, Tuple, Any, List
from phasespace import GenParticle, nbody_decay
from phasespace.fromdecay import GenMultiDecay
from decaylanguage import DecFileParser, DecayChainViewer 
from pathlib import Path
import torch


def gen_decay_from_file(
    decaylanguage: Dict[str, Any]
) -> Dict[Dict, Tuple[List, List]]:
    MOTHER_PARTICLE = decaylanguage["MOTHER_PARTICLE"] if "MOTHER_PARTICLE" in decaylanguage else None
    STABLE_PARTICLES = decaylanguage["STABLE_PARTICLES"] if "STABLE_PARTICLES" in decaylanguage else ()
    DECAY_FILE = decaylanguage["DECAY_FILE"] if "DECAY_FILE" in decaylanguage else None
    N_EVENTS = decaylanguage["N_EVENTS"] if "N_EVENTS" in decaylanguage else None
    VIEW_GRAPH = decaylanguage["VIEW_GRAPH"] if "VIEW_GRAPH" in decaylanguage else None

    parser = DecFileParser(DECAY_FILE)
    parser.parse()

    decay_chain = parser.build_decay_chains(MOTHER_PARTICLE, stable_particles=STABLE_PARTICLES)

    dcv = DecayChainViewer(decay_chain)
    dcv.graph.render(filename='decayGraph', format='pdf', view=VIEW_GRAPH, cleanup=True)

    decay_process = GenMultiDecay.from_dict(decay_chain)

    weights, events = decay_process.generate(n_events=N_EVENTS)


    return {
        "decay_tree_structure": decay_chain,
        "decay_tree_events": (weights, events)
    }


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



def gen_nbody_decay_data(
    parameters: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:

    particles = dict()

    MASSES = parameters["MASSES"] if "MASSES" in parameters else None
    FSP_MASSES = parameters["FSP_MASSES"] if "FSP_MASSES" in parameters else None
    N_TOPOLOGIES = parameters["N_TOPOLOGIES"] if "N_TOPOLOGIES" in parameters else None
    MAX_DEPTH = parameters["MAX_DEPTH"] if "MAX_DEPTH" in parameters else None
    MAX_CHILDREN = parameters["MAX_CHILDREN"] if "MAX_CHILDREN" in parameters else None
    MIN_CHILDREN = parameters["MIN_CHILDREN"] if "MIN_CHILDREN" in parameters else None
    ISP_WEIGHT = parameters["ISP_WEIGHT"] if "ISP_WEIGHT" in parameters else None
    TRAIN_EVENTS_PER_TOP = parameters["TRAIN_EVENTS_PER_TOP"] if "TRAIN_EVENTS_PER_TOP" in parameters else None
    VAL_EVENTS_PER_TOP = parameters["VAL_EVENTS_PER_TOP"] if "VAL_EVENTS_PER_TOP" in parameters else None
    TEST_EVENTS_PER_TOP = parameters["TEST_EVENTS_PER_TOP"] if "TEST_EVENTS_PER_TOP" in parameters else None
    SEED = parameters["SEED"] if "SEED" in parameters else None
    ISO_RETRIES = parameters["ISO_RETRIES"] if "ISO_RETRIES" in parameters else None
    GENERATE_UNKNOWN = parameters["GENERATE_UNKNOWN"] if "GENERATE_UNKNOWN" in parameters else None

    topologies = generate_phasespace(   MASSES,
                                        FSP_MASSES, 
                                        N_TOPOLOGIES, 
                                        MAX_DEPTH, 
                                        MAX_CHILDREN, 
                                        MIN_CHILDREN,
                                        ISP_WEIGHT,
                                        TRAIN_EVENTS_PER_TOP,
                                        VAL_EVENTS_PER_TOP,
                                        TEST_EVENTS_PER_TOP,
                                        SEED,
                                        ISO_RETRIES,
                                        GENERATE_UNKNOWN)
    
    gen_train_data(parameters, topologies)

    return {
        "parameters": parameters,
        "topologies": topologies
    }




def generate_phasespace(
        masses,
        fsp_masses,
        n_topologies=10,
        max_depth=5,
        max_children=6,
        min_children=2,
        isp_weight=1.,
        train_events_per_top=5,
        val_events_per_top=10,
        test_events_per_top=10,
        seed=None,
        iso_retries=0,
        generate_unknown=True,
):
    """ Generate a PhaseSpace dataset

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

    from phasespace import GenParticle
    # import tensorflow as tf

    if seed is not None:
        np.random.seed(seed)
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
        raise ValueError("Particles are only identified by their masses. Final state particle masses can not occur in intermediate particle masses.")


    topology_isomorphism_invariates = []

    total_topologies = n_topologies
    if generate_unknown:
        total_topologies = 3 * n_topologies

    topologies = list()
    
    for i in range(total_topologies):
        # NOTE generate tree for a topology
        for j in range(max(1, iso_retries)):
            queue = []
            root_node = GenParticle('root', masses[0])
            queue.append((root_node, 1))
            name = 1
            next_level = 1
            while len(queue) > 0:
                node, level = queue.pop(0)
                if next_level <= level:
                    next_level = level + 1
                num_children = np.random.randint(min_children, max_children + 1)

                total_child_mass = 0
                children = []

                # Mass we have to play with
                avail_mass = node._mass_val
                # Add an insurance to make sure it's actually possible to generate two children
                if avail_mass <= (2 * min(fsp_masses)):
                    raise ValueError("Any ISP mass given has to be larger than two times the smallest FSP mass.")

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
                        or np.random.random() < (1. * len(fsp_masses)) / ((1. * len(fsp_masses)) + (isp_weight * len(masses)))
                    ):
                        child_mass = np.random.choice([n for n in fsp_masses if (n < avail_mass)])
                    else:
                        child_mass = np.random.choice([n for n in masses if (n < avail_mass)])
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
                raise RuntimeError("Could not find sufficient number of non-isomorphic topologies.")
                # print("Could not find sufficient number of non-isomorphic topologies.")
                # continue

        topologies.append(root_node)


    return topologies

def count_leaves(node,):
    num_leaves = 0
    if node.has_grandchildren:
        for child in node.children:
            num_leaves += count_leaves(child)
    elif node.has_children:
        num_leaves += len(node.children)
    else:
        num_leaves += 1

    return num_leaves

def gen_train_data(parameters, topologies):
    N_TOPOLOGIES = parameters["N_TOPOLOGIES"] if "N_TOPOLOGIES" in parameters else None
    MODES_NAMES = parameters["MODES_NAMES"] if "MODES_NAMES" in parameters else None
    TRAIN_EVENTS_PER_TOP = parameters["TRAIN_EVENTS_PER_TOP"] if "TRAIN_EVENTS_PER_TOP" in parameters else None
    VAL_EVENTS_PER_TOP = parameters["VAL_EVENTS_PER_TOP"] if "VAL_EVENTS_PER_TOP" in parameters else None
    TEST_EVENTS_PER_TOP = parameters["TEST_EVENTS_PER_TOP"] if "TEST_EVENTS_PER_TOP" in parameters else None
    GENERATE_UNKNOWN = parameters["GENERATE_UNKNOWN"] if "GENERATE_UNKNOWN" in parameters else None

    events_per_mode = {'train': TRAIN_EVENTS_PER_TOP, 'val': VAL_EVENTS_PER_TOP, 'test': TEST_EVENTS_PER_TOP}

    all_lca = {mode:list() for mode in MODES_NAMES}
    all_leave = {mode:list() for mode in MODES_NAMES}
    all_weights = {mode:list() for mode in MODES_NAMES}
    all_events = {mode:list() for mode in MODES_NAMES}

    for i, root_node in enumerate(topologies):
        # NOTE generate leaves and labels for training, validation, and testing
        modes = []
        # For topologies not in the training set, save them to a different subdir
        # save_dir = Path(root, 'unknown')
        if i < N_TOPOLOGIES or not GENERATE_UNKNOWN:
            modes = MODES_NAMES
            # save_dir = Path(root, 'known')
        elif i < (2 * N_TOPOLOGIES):
            modes = MODES_NAMES[1:]
        else:
            modes = MODES_NAMES[2:]
        # save_dir.mkdir(parents=True, exist_ok=True)

        lca, names = decay2lca(root_node)

        for mode in modes:
            num_events = events_per_mode[mode]
            weights, events = root_node.generate(
                num_events,
                seed=np.random.randint(np.iinfo(np.int32).max),
            )

            all_weights[mode].append(weights)
            all_events[mode].append(weights)


def convert_to_lca(parameters, topologies, decay_tree_events):

    
        leaves = np.asarray([events[name] for name in names])
        leaves = leaves.swapaxes(0, 1)
        assert leaves.shape == (num_events, count_leaves(root_node), 4)

        # NOTE shuffle leaves for each sample
        leaves, lca_shuffled = shuffle_leaves(leaves, lca)

        all_lca[mode].append(lca_shuffled)
        all_leave[mode].append(leaves)
            
        del lca
        del names

    return {
        "parameters": parameters,
        "all_lca": all_lca,
        "all_leave": all_leave,
        "decay_tree_events": (weights, events)
    }


def shuffle_leaves(leaves, lca):
    """
    leaves (torch.Tensor): tensor containing leaves of shape (num_samples. num_leaves, num_features)
    lca torch.Tensor): tensor containing lca matrix of simulated decay of shape (num_leaves, num_leaves)
    """
    assert leaves.shape[1] == lca.shape[0]
    assert leaves.shape[1] == lca.shape[1]
    d = lca.shape[1]

    shuff_leaves = np.zeros(leaves.shape)
    shuff_lca = np.zeros((leaves.shape[0], *(lca.shape)))

    for idx in np.arange(leaves.shape[0]):
        perms = np.random.permutation(d)
        shuff_leaves[idx] = leaves[idx, perms]
        shuff_lca[idx] = lca[perms][:, perms]

    return shuff_leaves, shuff_lca


def assign_parenthetical_weight_tuples(node):
    """
    Args:
        node(phasespace.GenParticle

    After A. Aho, J. Hopcroft, and J. Ullman The Design and Analysis of Computer Algorithms. Addison-Wesley Publishing Co., Reading, MA, 1974, pp. 84-85.
    Credits to Alexander Smal

    """
    if not node.has_children:
        return f'({node.get_mass()})'

    child_tuples = [assign_parenthetical_weight_tuples(c) for c in node.children]
    child_tuples.sort()
    child_tuples = ''.join(child_tuples)

    return f'({node.get_mass()}{child_tuples})'

def decay2lca(root):
    ''' Return the LCA matrix of a decay

    Args:
        root (phasespace.GenParticle): the root particle of a decay tree

    Returns:
        numpy.ndarray, list: the lca matrix of the decay and the list of
        GenParticle names corresponding to each row and column

    '''
    # NOTE find all leaf nodes
    node_list = [root]
    leaf_nodes = []
    parents = {root.name: None, }
    generations = {root.name: 0}
    while len(node_list)>0:
        node = node_list.pop(0)
        # NOTE since there is no get parent
        for c in node.children:
            if c.name in parents: raise(ValueError('Node names have to be unique!'))
            parents[c.name] = node
            generations[c.name] = generations[node.name]+1
        if len(node.children) == 0:
            leaf_nodes.append(node)
        else:
            node_list = [c for c in node.children] + node_list
    # NOTE init results
    names = [l.name for l in leaf_nodes]
    lca_mat = np.zeros((len(leaf_nodes), len(leaf_nodes)), dtype=int)

    # NOTE fix skipped generations such that leaves are all in the same one
    # NOTE and nodes can be "pulled down"
    depth = max({generations[name] for name in names})
    for name in names:
        generations[name] = depth
    node_list = [l for l in leaf_nodes]
    while len(node_list)>0:
        node = node_list.pop(0)
        parent = parents[node.name]
        if parent is not None:
            node_list.append(parent)
            if len(node.children)>0:
                generations[node.name] = min({generations[n.name] for n in node.children})-1

    # NOTE trace ancestry for all leaves to root
    for i in range(len(leaf_nodes)):
        for j in range(i+1, len(leaf_nodes)):
            _lca = _find_lca(leaf_nodes[i], leaf_nodes[j], parents, generations)
            lca_mat[i, j] = _lca
            lca_mat[j, i] = _lca

    return lca_mat, names

def _find_lca(node1, node2, parents, generations):
    if node1.name == node2.name:
        raise ValueError("{} and {} have to be different.".format(node1, node2))
    ancestry1 = []
    parent = node1
    while parent is not None:
        ancestry1.append(parent)
        parent = parents[parent.name]

    ancestry2 = []
    parent = node2
    while parent is not None:
        ancestry2.append(parent)
        parent = parents[parent.name]

    ancestry1.reverse()
    ancestry2.reverse()

    # NOTE basically find common subroot
    for i, x in enumerate(ancestry1):
        if x.name != ancestry2[i].name:
            subroot = parents[x.name]
            g = generations[subroot.name]
            return generations[node1.name] - generations[subroot.name]

    return 0