import numpy as np


def set_case(case="B", mie=True):
    config = {}

    # physical constants
    config["eps_ext"] = 8.8542 * 10 ** (-12)
    config["mu_ext"] = 1.2566 * 10 ** (-6)
    config["c"] = 1 / np.sqrt(config["mu_ext"] * config["eps_ext"])

    # parameters
    # MIE
    if case == "A":
        config["k_ext"] = 3.0
        config["eps_rel"] = 2.1
        config["mu_rel"] = 1.0        

    elif case == "B":
        config["k_ext"] = 5.0
        config["eps_rel"] = 1.9
        config["mu_rel"] = 1.0

    # Teflon
    # config['k_ext'] = 1.047197551196598
    # config['eps_rel'] = 2.1
    # config['mu_rel'] = 1.

    config["k_int"] = config["k_ext"] * np.sqrt(config["eps_rel"] * config["mu_rel"])
    config["mu_int"] = config["mu_ext"] * config["mu_rel"]
    
    config["lambda"] = 2 * np.pi / config["k_ext"]
    config["frequency"] = config["k_ext"] * config["c"] / 2.0 / np.pi
    
    config["eta_rel"] = np.sqrt(config["mu_rel"] / config["eps_rel"])

    # Ferrite
    # config['k_ext'] =
    # config['eps_rel'] = 2.5
    # config['mu_rel'] = 1.6
    config["osrc"] = False
    
    if mie:
        # for incident_z
        config["polarization"] = np.array([0.0, 0.0, 1.0])
        config["direction"] = np.array([1.0, 0.0, 0.0], dtype="float64")
        config["osrc"] = True
    else:
        config["polarization"] = np.array([1.0 + 1j, 2.0, -1.0 - 1.0 / 3.0 * 1j])
        config["direction"] = np.array(
            [1.0 / np.sqrt(14), 2.0 / np.sqrt(14), 3.0 / np.sqrt(14)], dtype="float64"
        )
        
    # options for the Far Field at z=0
    config["number_of_angles"] = 400
    if mie:
        config["number_of_angles"] = 3601

    config["angles"] = np.pi * np.linspace(0, 2, config["number_of_angles"])
    config["far_field_points"] = np.array(
        [
            np.cos(config["angles"]),
            np.sin(config["angles"]),
            np.zeros(config["number_of_angles"]),
        ]
    )

    return config


config = set_case()
