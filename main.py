import os
import numpy as np
import subprocess
import multiprocessing as mp
from functools import partial


def create_file(num_ave,
                network_type,
                network_parameters,
                heterogeneous_hev_susceptibilities,
                heterogeneous_phev_susceptibilities,
                heterogeneous_bev_susceptibilities,
                heterogeneous_driving_patterns,
                alpha_phev,
                alpha_bev,
                advertised_type,
                h,
                time_horizon,
                folder_name):
    """
    runs a single simulation of the model
    """
    h_index, h = h
    file_name = folder_name + f"/{h_index}/sim-{num_ave}.txt"

    h_hev = 0
    h_phev = 0
    h_bev = 0

    if advertised_type == 'hev':
        h_hev = h
    elif advertised_type == 'phev':
        h_phev = h
    elif advertised_type == 'bev':
        h_bev = h

    if network_type == "SL":
        (L,) = network_parameters
        subprocess.run(
            f"alternative_fuel_vehicle_abm.exe {network_type} {heterogeneous_hev_susceptibilities} {heterogeneous_phev_susceptibilities} {heterogeneous_bev_susceptibilities} {heterogeneous_driving_patterns} {alpha_phev} {alpha_bev} {h_hev} {h_phev} {h_bev} {time_horizon} {file_name} {L}")
    elif network_type == "WS":
        (N, k, beta) = network_parameters
        subprocess.run(
            f"alternative_fuel_vehicle_abm.exe {network_type} {heterogeneous_hev_susceptibilities} {heterogeneous_phev_susceptibilities} {heterogeneous_bev_susceptibilities} {heterogeneous_driving_patterns} {alpha_phev} {alpha_bev} {h_hev} {h_phev} {h_bev} {time_horizon} {file_name} {N} {k} {beta}")
        return 0


if __name__ == '__main__':
    # this script runs simulations that are used to create diagrams illustrating how the change of the marketing
    # strength for a given vehicle type (HEV, PHEV, or BEV) impacts the adoption levels of different types of AFVs
    # for a chosen set of parameters

    # choose the network type: "SL" - square lattice, "WS" Watts Strogatz network
    network_type = "WS"

    # choose whether susceptibilities and driving patterns are heterogeneous (set to 1) or homogeneous (set to 0)
    heterogeneous_hev_susceptibilities = 1
    heterogeneous_phev_susceptibilities = 1
    heterogeneous_bev_susceptibilities = 1
    heterogeneous_driving_patterns = 1

    # choose calibration parameters that show up in the formula for the refueling effect (RFE), see Eq.(5)
    alpha_phev = 8
    alpha_bev = 0.6

    # choose the time length of the simulation in Monte Carlo steps (MCS)
    time_horizon = 50

    # choose the number of independent simulations to carry out with the chosen parameters
    num_ave = 100

    ave_tab = np.arange(num_ave)

    # choose the advertised typ of car: "hev", "phev", or "bev"
    # only the chosen type of AFVs is advertised, the marketing strengths of the remaining car types are set to 0
    advertised_type = "hev"

    # creating folder name
    if network_type == "SL":
        # choose the system size:
        L = 50  # linear system size (N = L x L: number of agents)
        network_parameters = (L,)
        folder_name = f'{network_type}_{L}L_{heterogeneous_hev_susceptibilities}_{heterogeneous_phev_susceptibilities}_{heterogeneous_bev_susceptibilities}hs_{heterogeneous_driving_patterns}hdp_{alpha_phev}aphev_{alpha_bev}abev_{advertised_type}'
    elif network_type == "WS":
        # choose the system size:
        N = 2500  # number of agents
        # choose the network parameters, see Section 4.1. Social network:
        k = 4  # average node degree (must be divisible by 2)
        beta = 1  # rewiring probability
        network_parameters = (N, k, beta)
        folder_name = f'{network_type}_{N}N_{k}k_{beta}beta_{heterogeneous_hev_susceptibilities}_{heterogeneous_phev_susceptibilities}_{heterogeneous_bev_susceptibilities}hs_{heterogeneous_driving_patterns}hdp_{alpha_phev}aphev_{alpha_bev}abev_{advertised_type}'

    # choose the values of the marketing strength for the simulations
    num_h = 20  # number of considered values in the interval
    h_start = 0  # lower limit of the interval
    h_end = 4   # upper limit of the interval

    h_indexes = np.arange(num_h)
    h = np.linspace(h_start, h_end, num_h)

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    h_tuples = []
    for i in h_indexes:
        h_tuples.append((i, h[i]))
        if not os.path.exists(folder_name + f'/{i}'):
            os.mkdir(folder_name + f'/{i}')

    np.savetxt(folder_name + '/num_ave.csv', [num_ave], fmt="%i")
    np.savetxt(folder_name + '/calibration_params.csv', np.column_stack([alpha_phev, alpha_bev]), fmt=("%.3f", "%.3f"))
    np.savetxt(folder_name + '/h_values.csv', np.column_stack((h_indexes, h)), fmt=("%i", "%.18f"))

    for item in h_tuples:
        with mp.Pool(6) as pool:
            pool.map(partial(create_file,
                             network_type=network_type,
                             network_parameters=network_parameters,
                             heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                             heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                             heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                             heterogeneous_driving_patterns=heterogeneous_driving_patterns,
                             alpha_phev=alpha_phev,
                             alpha_bev=alpha_bev,
                             advertised_type=advertised_type,
                             h=item,
                             time_horizon=time_horizon,
                             folder_name=folder_name),
                     ave_tab)
