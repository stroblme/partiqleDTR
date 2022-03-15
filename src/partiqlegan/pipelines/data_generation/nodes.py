"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.17.7
"""
import numpy as np
from typing import Dict, Tuple
from phasespace import GenParticle, nbody_decay
def gen_decay_from_file(
    parameters: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    MOTHER_PARTICLE = parameters["MOTHER_PARTICLE"] if "MOTHER_PARTICLE" in parameters else None
    DECAY_FILE = parameters["DECAY_FILE"] if "DECAY_FILE" in parameters else None
    N_EVENTS = parameters["N_EVENTS"] if "N_EVENTS" in parameters else None

    parser = DecFileParser(DECAY_FILE)
    parser.parse()

    chain = parser.build_decay_chains(MOTHER_PARTICLE)
    DecayChainViewer(chain)

    decay_process = GenMultiDecay.from_dict(chain)

    weights, events = decay_process.generate(n_events=N_EVENTS)


    return events


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
    weights, tree = particles["Bp_0"].generate(n_events=N_EVENTS)

    for i, p in enumerate(tree):
        tree[p] = np.array(tree[p]).reshape(4,N_EVENTS)

    return tree