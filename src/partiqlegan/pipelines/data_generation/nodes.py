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


def gen_nbody_decay_data(
    parameters: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:

    particles = dict()
    N_EVENTS = parameters["N_EVENTS"] if "N_EVENTS" in parameters else None


    # Retrive particle masses and number of events
    for key, value in parameters.items():
        if "MASS" in key:
            particleName = key.replace("_MASS","")
            particles[f"{particleName}_0"] = GenParticle(particleName, value)

    # Add some extra particles
    particles["Pp_1"] = GenParticle("Pp_1", parameters["Pp_MASS"])
    particles["Pm_1"] = GenParticle("Pm_1", parameters["Pm_MASS"])
    particles["P0_1"] = GenParticle("P0_1", parameters["P0_MASS"])

    # Build the decay tree
    particles["D0_0"].set_children(particles["Kp_0"], particles["Pm_0"], particles["P0_0"])
    particles["O_0"].set_children(particles["Pp_1"], particles["Pm_1"], particles["P0_1"])
    particles["Bp_0"].set_children(particles["D0_0"], particles["O_0"], particles["Pp_0"])

    # Generate a few events
    weights, events = particles["Bp_0"].generate(n_events=N_EVENTS)

    for i, p in enumerate(events):
        events[p] = np.array(events[p]).reshape(4,N_EVENTS)

    return {
        "decay_tree_structure": decay_chain,
        "decay_tree_events": (weights, events)
    }

class TreeSet(torch.utils.data.Dataset):
    """ Dataset holding trees to feed to network"""
    def __init__(self, x, y):
        """ In our use x will be the array of leaf attributes and y the LCA matrix, i.e. the labels"""
        self.x = x
        self.y = y
        return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.long))


class PhasespaceSet(TreeSet):
    def __init__(
        self,
        root,
        mode='train',
        file_ids=None,
        samples=None,
        samples_per_topology=None,
        seed=None,
        apply_scaling=False,
        scaling_dict=None,
        **kwargs,
    ):
        """ Dataset handler thingy for data generated with PhaseSpace

        Args:
            root(str or Path): the root folder containing all files belonging to the dataset in its different modes or partitions
            mode(str): 'train', 'val' or 'test' mode
            file_id(int or list(int)): Optional identifier or list of identifiers of files to load
            samples(int): number of samples to load total, will be a random subset. If larger than the samples in loaded files then this is ignored
            samples_per_topology(int): number of samples to load from each file, will be a random subset. If larger than the samples in loaded files then this is ignored
            seed(int): Random seed to use for selecting sample subset
            apply_scaling(bool): Whether to apply standard scaling to features, scaling factors will be calculated if scaling_dict is None.
            scaling_dict(dict): Scaling factors contained as {mean: float, std: float}
        """
        self.root = Path(root)

        self.modes = ['train', 'val', 'test']
        if mode not in self.modes:
            raise ValueError("unknown mode")

        self.mode = mode
        # self.known = known
        self.apply_scaling = apply_scaling
        self.scaling_dict = scaling_dict

        if seed is not None:
            np.random.seed(seed)

        x_files = sorted(self.root.glob(f'leaves_{mode}.*'))
        y_files = sorted(self.root.glob(f'lcas_{mode}.*'))

        assert len(x_files) > 0, f"No files to load found in {self.root}"

        # This deals with 0 padding dataset IDs since python will convert the input numbers to ints
        # Assumes files are save as xxxxx.<file_id>.npy
        if file_ids is not None:
            # Make sure file_ids is a list of ints
            file_ids = [int(file_ids)] if not isinstance(file_ids, list) else [int(i) for i in file_ids]

            x_files = [x for x in x_files if int(x.suffixes[0][1:]) in file_ids]
            y_files = [y for y in y_files if int(y.suffixes[0][1:]) in file_ids]

        # # In the case we're not loading train files, need to separate the known topologies from the unknown
        # # This is very hackish, really should have place the un/known files in separate directories to begin with
        # if self.mode is not 'train':
        #     train_files = sorted(self.root.glob('leaves_train.*'))
        #     train_files = [x.suffixes[0] for x in train_files]

        #     if self.known:
        #         x_files = [x for x in x_files if x.suffixes[0] in train_files]
        #         y_files = [y for y in y_files if y.suffixes[0] in train_files]
        #     else:
        #         x_files = [x for x in x_files if not x.suffixes[0] in train_files]
        #         y_files = [y for y in y_files if not y.suffixes[0] in train_files]

        if len(x_files) != len(y_files):
            raise RuntimeError(f'"leaves" and "lcas" files in {self.root} don\'t match')

        self.x = [np.load(f) for f in x_files]
        self.y = [np.load(f) for f in y_files]

        # Don't assume same number of sample per topology
        self.samples_per_topology = np.array([a.shape[0] for a in self.x])

        # Select a random subset of samples from each topology
        if samples_per_topology is not None and samples_per_topology < min(self.samples_per_topology):
            # Choose which samples to take from each file
            ids = [np.random.choice(i, samples_per_topology, replace=False) for i in self.samples_per_topology]
            self.x = [f[ids[i]] for i, f in enumerate(self.x)]
            self.y = [f[ids[i]] for i, f in enumerate(self.y)]
            # Set a fixed number of samples per topology, could be an int but makes things messier later :(
            self.samples_per_topology = np.array([samples_per_topology] * len(self.x))

        # Need this to know which files to take samples from
        self.cum_samples = np.cumsum(self.samples_per_topology)
        # And this to know where in the files
        self.loc_samples = self.cum_samples - self.samples_per_topology

        # Intentionally selecting one subset of indexes for all files so it's reproducible
        # even if only a subset of the files are loaded
        # TODO: Change this to still keep x a list of topology arrays, just with differing lengths
        if samples is not None and samples < sum(self.samples_per_topology):
            # Need this to know which files to take samples from
            cum_samples = np.cumsum(self.samples_per_topology)
            # And this to know where in the files
            loc_samples = cum_samples - self.samples_per_topology

            ids = np.random.choice(cum_samples[-1], samples, replace=False)
            file_ids = np.searchsorted(cum_samples, ids, side='left')
            # Get the true ids locations in each file
            ids = ids - loc_samples[file_ids]
            # self.x = [arr[idx] for arr in self.x]
            # self.y = [arr[idx] for arr in self.y]
            # This is a lazy way to avoid more intelligently extracting the correct item in getitem below
            # It just pretends there's one sample per topology and (samples) number of topologies
            self.x = [self.x[f][i] for f, i in zip(file_ids, ids)]
            self.y = [self.y[f][i] for f, i in zip(file_ids, ids)]

            # Selecting a subset we have a list of individual topolgies
            self.samples_per_topology = 1

        # If scaling is requested, check the scaling factors exist and calculate them if not
        if apply_scaling and self.scaling_dict is None:
            # Calculate values from 10% of the data
            self.scaling_dict = self._calculate_scaling_dict(int(self.__len__() * 0.1))

    def _calculate_scaling_dict(self, n_samples=1000):
        ''' Calculate scalings to bring features around the [-1, 1] range.

        This calculates a standard normalisation, i.e.:
            (x - mean)/std

        Args:
            n_samples (int, optional): Number of samples to use when calculating scalings
        Returns:
            Scaling dictionary containing {mean, std}  arrays of values for each feature
        '''
        # Select a random subset to calculate scalings from
        # In this case treating all samples as equal, so flatten them as if it's one long list of detected particles
        # Alternative approach would be to calculate the mean/std of each sample individually, then calculate their means
        x_sample = np.concatenate([self.__getitem__(i)[0] for i in np.random.choice(self.__len__(), size=n_samples, replace=False)])  # (n_samples*l, d)
        mean = np.mean(x_sample, axis=0)  # (d,)
        std = np.std(x_sample, axis=0)  # (d,)

        return {'mean': mean, 'std': std}

    def __getitem__(self, idx):
        ''' This currently has two modes:
            1. When self.x is still a list of one array per file (the if clause below)
            2. When self.x is a list of individual samples (the else clause below)
        '''
        idx = int(idx)
        if isinstance(self.samples_per_topology, np.ndarray):
            # Find file and location of this sample
            file_idx = np.searchsorted(self.cum_samples, idx, side='right')
            idx = idx - self.loc_samples[file_idx]
            item = [self.x[file_idx][idx], self.y[file_idx][idx]]
        else:
            item = [self.x[idx], self.y[idx]]

        # Apply scaling dict if requested
        if self.apply_scaling and self.scaling_dict is not None:
            item[0] = (item[0] - self.scaling_dict['mean']) / self.scaling_dict['std']

        # Set diagonal to -1, our padding must be -1 as well so we can tell the Loss to ignore it
        np.fill_diagonal(item[1], -1)

        return (
            torch.tensor(item[0], dtype=torch.float),  # (l, d)
            torch.tensor(item[1], dtype=torch.long)  # (l, l)  NOTE: confirm this
        )

    def __len__(self):
        # Handle case that we have selected samples randomly
        return self.samples_per_topology.sum() if isinstance(self.samples_per_topology, np.ndarray) else len(self.x)
