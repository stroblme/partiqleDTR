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
) -> Tuple[Dict[str, np.ndarray]]:
    B0_MASS = parameters["B0_MASS"] if "B0_MASS" in parameters else None
    KSTARZ_MASS = parameters["KSTARZ_MASS"] if "KSTARZ_MASS" in parameters else None
    PION_MASS = parameters["PION_MASS"] if "PION_MASS" in parameters else None
    KAON_MASS = parameters["KAON_MASS"] if "KAON_MASS" in parameters else None
    N_EVENTS  = parameters["N_EVENTS"] if "N_EVENTS" in parameters else 1000

    weights, particles = nbody_decay(B0_MASS, [PION_MASS, KAON_MASS]).generate(n_events=N_EVENTS)

    result = pd.DataFrame(weights, particles)
    return result


def gen_sequential_decay_data(
    parameters: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray]]:
    B0_MASS = parameters["B0_MASS"] if "B0_MASS" in parameters else None
    KSTARZ_MASS = parameters["KSTARZ_MASS"] if "KSTARZ_MASS" in parameters else None
    PION_MASS = parameters["PION_MASS"] if "PION_MASS" in parameters else None
    KAON_MASS = parameters["KAON_MASS"] if "KAON_MASS" in parameters else None
    N_EVENTS  = parameters["N_EVENTS"] if "N_EVENTS" in parameters else 1000

    kaon = GenParticle('K+', KAON_MASS)
    pion = GenParticle('pi-', PION_MASS)
    kstar = GenParticle('K*', KSTARZ_MASS).set_children(kaon, pion)
    gamma = GenParticle('gamma', 0)
    bz = GenParticle('B0', B0_MASS).set_children(kstar, gamma)

    weights, particles = bz.generate(n_events=N_EVENTS)

    return weights, particles