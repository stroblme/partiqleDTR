"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.17.7
"""
import numpy as np
from typing import Dict, Tuple, Any, List
from phasespace import GenParticle, nbody_decay
from phasespace.fromdecay import GenMultiDecay
from decaylanguage import DecFileParser, DecayChainViewer 


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
#     weights, tree = particles["Bp_0"].generate(n_events=N_EVENTS)

#     for i, p in enumerate(tree):
#         tree[p] = np.array(tree[p]).reshape(4,N_EVENTS)

#     return tree