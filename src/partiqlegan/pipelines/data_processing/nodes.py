"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""
import numpy as np
from typing import Dict, Tuple, List
from sqlalchemy import Float, desc

# import torch
import torchvision
import torchdata as td
from torch import LongTensor, FloatTensor, cat
from torch.utils.data import Dataset
import copy

import re

data_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# class DecayTreeDataset(torch.utils.data.dataset):
#     def __init__(self, dataset:Dict[str, np.ndarray]):

#         self.decayTreeDataset = dataset
        
#         raise NotImplementedError("Sorry, not yet..")

#     def __len__(self):
#         return len(self.decayTreeDataset)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         raise NotImplementedError("Sorry, not yet..")
        
allowed_fsps = ["e+", "e-", "pi+", "pi-", "gamma"]
allowed_nodes = ["e+", "e-", "pi+", "pi-", "pi0", "gamma", "omega", "mu+", "mu-", "eta"]

class particle_node():
    def __init__(self, name, parent_node=None, fsp=False):
        self.name = name
        self.parent_node = parent_node
        self.fsp = fsp

    # def __eq__(self, __o: object) -> bool:
    #     try:
    #         return (self.name == __o.name) and (self.parent_node == __o.parent_node)
    #     except:
    #         return False

def tree_data_to_adjacency_list(
    decay_tree_structure: Tuple[List, List]
) -> Tuple[List, List]:
    particles = list()
    adjacency_list = list()

    def append_node(list_to_append, node, parent_node_instance, fsp=False):
        if type(node) == str:
            if node in allowed_nodes:
                particle_instance = particle_node(node, parent_node_instance, fsp=fsp)
            else:
                print(f"{node} not in allowed_nodes")
        else:
            particle_instance = node

        list_to_append.append(particle_instance)
        particles.append(particle_instance)
        return particle_instance

    def append_list(list_to_append, other_list):
        list_to_append.append(other_list)
        return list_to_append[-1]
    

    def create_adj_list_from_tree(tree, adj_list, parent_node=particle_node(name="origin")):

        first_key = list(tree.keys())[0]

        # fsps
        if "fs" and "bf" in tree.keys():
            # First, add a new list to which we will append any subsequent leaves
            adj_list[-1].append(list())

            # strings are moved to the very beginning. This is important to not mixup with the [-1] calls
            type_sorted_values=sorted(tree["fs"], key=lambda x: not isinstance(x,str))
            for node_name in type_sorted_values:
                # A string? add it directly to the list
                if type(node_name) == str:
                    append_node(adj_list[-1][-1], node_name, parent_node, fsp=True)
                # Special treating for the dicts
                else:
                    # these are only 1 dim dicts, thus it's fair enough to just get the first key
                    node = append_node(adj_list[-1][-1], list(node_name.keys())[0], parent_node)
                    # a recursive call at this point is ok, since there should be no strings anymore due to the sorting prior to the iteration
                    create_adj_list_from_tree(node_name, adj_list, node)
        # key is a node
        elif first_key in allowed_nodes:
            # store the key in a more meaningful variable. This will become relevant in a few lines
            node = first_key
            for entry in tree[first_key]:
                new_list = append_list(adj_list, list())
                # careful here, in the first run, node equals a string, in the second, it becomes the node instance as
                # the variable is overwritten at this certain line.
                # append_node allows either string or a node instance and thus either reuses or creates a new node instance
                node = append_node(new_list, node, parent_node.parent_node)
                create_adj_list_from_tree(entry, adj_list, node)

    create_adj_list_from_tree(decay_tree_structure, adjacency_list)

    return (particles, adjacency_list)

def adjacency_list_to_adjacency_matrix(
    particles: List,
    adjacency_list: List
) -> np.ndarray:
    adjacency_matrix = np.zeros((len(particles), len(particles)))

    def create_adj_mat_from_adj_list(adj_list, adj_matrix):
        np.fill_diagonal(adj_matrix, 1)
        
        for edge in adj_list:
            idx_a = particles.index(edge[0])
            for set_of_childs in edge[1:]:
                for child in set_of_childs:
                    idx_b = particles.index(child)
                    adj_matrix[idx_a, idx_b] += 1
                    adj_matrix[idx_b, idx_a] += 1


    create_adj_mat_from_adj_list(adjacency_list, adjacency_matrix)
    
    return adjacency_matrix

def filter_fsps(
    particles: List
) -> List:
    fsps = [particle for particle in particles if particle.fsp]

    return fsps
    

def adjacency_list_to_lca(
    particles: List,
    adjacency_list: List
) -> np.ndarray:
    
    fsps = filter_fsps(particles)

def adjacency_list_to_lcas(
    particles: List,
    adjacency_list: List
) -> np.ndarray:
    pass

def adjacency_list_to_lcag(
    particles: List,
    adjacency_list: List
) -> np.ndarray:
    pass




def conv_structure_to_lca_and_names(pad_to:int, decay_tree_structure):
    all_lca = list()
    all_names = list()

    for i, root_node in enumerate(decay_tree_structure):
        lca, names = _conv_decay_to_lca(root_node, pad_to)

        all_lca.append(lca)
        all_names.append(names)
                

    return {
        "all_lca": all_lca,
        "all_names": all_names
    }


def lca_and_leaves_sort_into_modes(
    n_topologies: int,
    modes_names: List[str],
    train_events_per_top: int,
    val_events_per_top: int,
    test_events_per_top: int,
    generate_unknown: bool,
    all_lca: List,
    all_names: List,
    decay_tree_events: Tuple[List, List]):
    
    # n_topologies = decay_parameters["N_TOPOLOGIES"] if "N_TOPOLOGIES" in decay_parameters else None
    # modes_names = decay_parameters["MODES_NAMES"] if "MODES_NAMES" in decay_parameters else None
    # train_events_per_top = decay_parameters["TRAIN_EVENTS_PER_TOP"] if "TRAIN_EVENTS_PER_TOP" in decay_parameters else None
    # val_events_per_top = decay_parameters["VAL_EVENTS_PER_TOP"] if "VAL_EVENTS_PER_TOP" in decay_parameters else None
    # test_events_per_top = decay_parameters["TEST_EVENTS_PER_TOP"] if "TEST_EVENTS_PER_TOP" in decay_parameters else None
    # generate_unknown = decay_parameters["GENERATE_UNKNOWN"] if "GENERATE_UNKNOWN" in decay_parameters else None

    events_per_mode = {'train': train_events_per_top, 'val': val_events_per_top, 'test': test_events_per_top}

    _, all_events = decay_tree_events

    all_lca_mode_sorted = {mode:list() for mode in modes_names}
    all_leaves_mode_sorted = {mode:list() for mode in modes_names}

    for i, (lca, names) in enumerate(zip(all_lca, all_names)):
        # NOTE generate leaves and labels for training, validation, and testing
        modes = []
        # For topologies not in the training set, save them to a different subdir
        # save_dir = Path(root, 'unknown')
        if i < n_topologies or not generate_unknown:
            modes = modes_names
            # save_dir = Path(root, 'known')
        elif i < (2 * n_topologies):
            modes = modes_names[1:]
        else:
            modes = modes_names[2:]
        # save_dir.mkdir(parents=True, exist_ok=True)


        for mode in modes:
            num_events = events_per_mode[mode]
    
            leaves = np.asarray([all_events[mode][i][name] if name in all_events[mode][i] else np.zeros((num_events, 4)) for name in names])
            leaves = leaves.swapaxes(0, 1)
            # assert leaves.shape == (num_events, _count_leaves(root_node), 4)

            # NOTE shuffle leaves for each sample
            # leaves, lca_shuffled = _shuffle_leaves(leaves, lca)

            all_lca_mode_sorted[mode].append(lca)
            all_leaves_mode_sorted[mode].append(leaves)
                

    return {
        "all_lca_mode_sorted": all_lca_mode_sorted,
        "all_leaves_mode_sorted": all_leaves_mode_sorted
    }

def shuffle_lca_and_leaves_in_mode(
    modes_names: List[str],
    all_lca_mode_sorted: List,
    all_leaves_mode_sorted: List):
    
    # modes_names = decay_parameters["MODES_NAMES"] if "MODES_NAMES" in decay_parameters else None

    all_lca_shuffled = {mode:list() for mode in modes_names}
    all_leaves_shuffled = {mode:list() for mode in modes_names}

    for mode in modes_names:

        for (lca, leaves) in zip(all_lca_mode_sorted[mode], all_leaves_mode_sorted[mode]):
            # NOTE shuffle leaves for each sample
            leaves_shuffled, lca_shuffled = _shuffle_lca_and_leaves(leaves, lca)

            all_lca_shuffled[mode].append(lca_shuffled)
            all_leaves_shuffled[mode].append(leaves_shuffled)
                

    return {
        "all_lca_shuffled": all_lca_shuffled,
        "all_leaves_shuffled": all_leaves_shuffled
    }

# def shuffle_lca_and_leaves(
#     decay_parameters: List,
#     all_lca: List,
#     all_names: List,
#     decay_tree_events: Tuple[List, List]):
    
#     N_TOPOLOGIES = decay_parameters["N_TOPOLOGIES"] if "N_TOPOLOGIES" in decay_parameters else None
#     MODES_NAMES = decay_parameters["MODES_NAMES"] if "MODES_NAMES" in decay_parameters else None
#     TRAIN_EVENTS_PER_TOP = decay_parameters["TRAIN_EVENTS_PER_TOP"] if "TRAIN_EVENTS_PER_TOP" in decay_parameters else None
#     VAL_EVENTS_PER_TOP = decay_parameters["VAL_EVENTS_PER_TOP"] if "VAL_EVENTS_PER_TOP" in decay_parameters else None
#     TEST_EVENTS_PER_TOP = decay_parameters["TEST_EVENTS_PER_TOP"] if "TEST_EVENTS_PER_TOP" in decay_parameters else None
#     GENERATE_UNKNOWN = decay_parameters["GENERATE_UNKNOWN"] if "GENERATE_UNKNOWN" in decay_parameters else None

#     events_per_mode = {'train': TRAIN_EVENTS_PER_TOP, 'val': VAL_EVENTS_PER_TOP, 'test': TEST_EVENTS_PER_TOP}

#     _, all_events = decay_tree_events

#     all_lca_shuffled = {mode:list() for mode in MODES_NAMES}
#     all_leaves_shuffled = {mode:list() for mode in MODES_NAMES}

#     for i, (lca, names) in enumerate(zip(all_lca, all_names)):
#         # NOTE generate leaves and labels for training, validation, and testing
#         modes = []
#         # For topologies not in the training set, save them to a different subdir
#         # save_dir = Path(root, 'unknown')
#         if i < N_TOPOLOGIES or not GENERATE_UNKNOWN:
#             modes = MODES_NAMES
#             # save_dir = Path(root, 'known')
#         elif i < (2 * N_TOPOLOGIES):
#             modes = MODES_NAMES[1:]
#         else:
#             modes = MODES_NAMES[2:]
#         # save_dir.mkdir(parents=True, exist_ok=True)


#         for mode in modes:
#             num_events = events_per_mode[mode]
    
#             leaves = np.asarray([all_events[mode][i][name] if name in all_events[mode][i] else np.zeros((num_events, 4)) for name in names])
#             leaves = leaves.swapaxes(0, 1)
#             # assert leaves.shape == (num_events, _count_leaves(root_node), 4)

#             # NOTE shuffle leaves for each sample
#             leaves_shuffled, lca_shuffled = _shuffle_lca_and_leaves(leaves, lca)

#             all_lca_shuffled[mode].append(lca_shuffled)
#             all_leaves_shuffled[mode].append(leaves_shuffled)
                

#     return {
#         "all_lca_shuffled": all_lca_shuffled,
#         "all_leaves_shuffled": all_leaves_shuffled
#     }

def _conv_decay_to_lca(root, pad_to=None):
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

    if pad_to != None:
        #increase pad_to parameter if an error occurs here
        lca_res = np.pad(lca_mat, [(0,pad_to-lca_mat.shape[0]), (0,pad_to-lca_mat.shape[1])], 'constant')
        names_res = np.pad(names, (0, pad_to-len(names)), 'constant')
        return lca_res, names_res

    return lca_mat, names

def _shuffle_lca_and_leaves(leaves, lca):
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


def _count_leaves(node,):
    num_leaves = 0
    if node.has_grandchildren:
        for child in node.children:
            num_leaves += _count_leaves(child)
    elif node.has_children:
        num_leaves += len(node.children)
    else:
        num_leaves += 1

    return num_leaves


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

class TreeSet(Dataset):
    """ Dataset holding trees to feed to network"""
    def __init__(self, x, y):
        """ In our use x will be the array of leaf attributes and y the LCA matrix, i.e. the labels"""
        self.x = x
        self.y = y
        return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (FloatTensor(self.x[idx]), LongTensor(self.y[idx]))


def lca_and_leaves_to_tuple_dataset(
    all_lca_shuffled, all_leaves_shuffled
) -> Dict[str, Tuple[List, List]]:

    modes = all_lca_shuffled.keys()
    dataset_lca_and_leaves = dict()

    for mode in modes:
        x_data = []
        y_data = []
        for topology_it in range(len(all_lca_shuffled[mode])):
            for i in range(len(all_lca_shuffled[mode][topology_it])):
                x_data.append(all_leaves_shuffled[mode][topology_it][i])
                y_data.append(all_lca_shuffled[mode][topology_it][i])

        dataset_lca_and_leaves[mode] = TreeSet(x_data, y_data)

    return {
        "dataset_lca_and_leaves":dataset_lca_and_leaves
    }

def tuple_dataset_to_torch_tensor_dataset(
    tuple_dataset: Dict[str, Tuple[List, List]]
) -> Dict[str, Tuple[LongTensor, FloatTensor]]:
    torch_dataset_lca_and_leaves = dict()

    # for key, value in tuple_dataset.items():
    #     values = []
    #     for v in value:
    #         v_0_accum.append(LongTensor(v))
    #     v_1_accum = []
    #     for v in value[1]:
    #         v_1_accum.append(FloatTensor(v))

    #     torch_dataset_lca_and_leaves[key] = (v_0_accum, v_1_accum)

    return {
        "torch_dataset_lca_and_leaves":tuple_dataset
    }


def tree_data_to_discriminator(
    decay_tree_structure: Tuple[List, List]
) -> Dict[str, np.ndarray]:

    # raise NotImplementedError("Sorry, not yet..")
    probabilities = list()
    particles, adjacency_list = tree_data_to_adjacency_list(decay_tree_structure)
    adjacency_matrix = adjacency_list_to_adjacency_matrix(particles, adjacency_list)
    lcas_matrix = adjacency_list_to_lcas()


def tree_data_to_generator(
    decay_tree_events: Dict
) -> Dict[str, np.ndarray]:

    # raise NotImplementedError("Sorry, not yet..")

    num_removed_particles = 0

    def combine_weights_and_particles(weights, particles):
        assert len(weights) == len(particles)

        nonlocal num_removed_particles
        num_of_sets = len(particles)
        empty_sets = list()

        for set_i in range(num_of_sets):
            particles_in_set = list(particles[set_i].keys())

            for particle in particles_in_set:
                try:
                    idx = [f in particle for f in allowed_fsps].index(True)
                except ValueError:
                    idx = None
                    particles[set_i].pop(particle)  #remove the particle from the set
                    num_removed_particles += 1

                if particles[set_i] == {}:
                    empty_sets.append(set_i)

        shifter = 0
        for set_i in empty_sets:
            particles.pop(set_i-shifter)
            weights.pop(set_i-shifter)
            shifter += 1

        print(f"Removed {num_removed_particles} particles from the events as they are not in the list of allowed fsps: {allowed_fsps}")

    combine_weights_and_particles(*decay_tree_events)

    return decay_tree_events


def normalize_event(
    decay_parameters: Dict,
    data: Dict
) -> Dict:

    MODES_NAMES = decay_parameters["MODES_NAMES"] if "MODES_NAMES" in decay_parameters else None

    for mode in MODES_NAMES:
        for i_topology in range(len(data[mode])):
            data[mode][i_topology] = data[mode][i_topology]/max(data[mode][i_topology].max(), abs(data[mode][i_topology].min()))

    return data


# def pad_lca(
#     decay_parameters: Dict,
#     data: Dict
# ) -> Dict:

#     MODES_NAMES = decay_parameters["MODES_NAMES"] if "MODES_NAMES" in decay_parameters else None
#     PAD_TO = decay_parameters["PAD_TO"] if "PAD_TO" in decay_parameters else None

#     for mode in MODES_NAMES:
#         for i in range(len(data[mode])):
#             for j in range(len(data[mode][i])):
#                 shape = data[mode][i][j].shape
#                 if shape[0] > PAD_TO:
#                     raise RuntimeError(f"Padding not sufficient. Is {PAD_TO} but requires {shape[0]}")
#                 elif shape[0] < PAD_TO:
#                     padded = np.pad(data[mode][i][j], [(0,PAD_TO-shape[0]), (0,PAD_TO-shape[1])], mode="constant")
#                     data[mode][i][j].resize((PAD_TO, PAD_TO), refcheck=False)
#                     data[mode][i][j] = padded
                

#     return data