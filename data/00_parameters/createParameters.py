import pickle

FOLDER_NAME = "./data/00_parameters/"
FILE_NAME = "phasespace.pkl"

parameters={
    "Bm_MASS":5279.34,
    "Bp_MASS":5279.34,
    "B0_MASS":5279.65,
    "Bs0_MASS":5366.88,
    "Bcp_MASS":6274.9,
    "Km_MASS":493.677,
    "Kp_MASS":493.677,
    "K0_MASS":497.611,
    "Dm_MASS":1869.65,
    "Dp_MASS":1869.65,
    "D0_MASS":1864.83,
    "Dsp_MASS":1968.34,
    "KSTARZ_MASS":895.55,
    "Pm_MASS":139.57039,
    "Pp_MASS":139.57039,
    "P0_MASS":134.9768,
    "O_MASS":782.65,
    "N_EVENTS":100
}

destinationFile = open(FOLDER_NAME+FILE_NAME, mode="wb")
pickle.dump(parameters, destinationFile)



FOLDER_NAME = "./data/00_parameters/"
FILE_NAME = "decaylanguage.pkl"

parameters={
    "MOTHER_PARTICLE":"pi0",
    "DECAY_FILE":"./data/01_raw/DECAY_LHCB.dec",
    "N_EVENTS":100
}

destinationFile = open(FOLDER_NAME+FILE_NAME, mode="wb")
pickle.dump(parameters, destinationFile)