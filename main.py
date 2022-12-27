import os
import numpy as np
import subprocess
import multiprocessing as mp
from functools import partial


def create_file(num_ave,
                network_type,
                network_parameters,
                heterogeneous_susceptibilities,
                heterogeneous_driving_patterns,
                alpha_phev,
                alpha_bev,
                advertised_type,
                h,
                time_horizon,
                folder_name):
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
            f"alternative_fuel_vehicle_abm.exe {network_type} {heterogeneous_susceptibilities} {heterogeneous_driving_patterns} {alpha_phev} {alpha_bev} {h_hev} {h_phev} {h_bev} {time_horizon} {file_name} {L}")
    elif network_type == "WS":
        (N, k, beta) = network_parameters
        subprocess.run(
            f"alternative_fuel_vehicle_abm.exe {network_type} {heterogeneous_susceptibilities} {heterogeneous_driving_patterns} {alpha_phev} {alpha_bev} {h_hev} {h_phev} {h_bev} {time_horizon} {file_name} {N} {k} {beta}")
        return 0


if __name__ == '__main__':

    network_type = "WS"  # "SL" - square lattice, "WS" Watts Strogatz network

    heterogeneous_susceptibilities = 1
    heterogeneous_driving_patterns = 1

    alpha_phev = 8
    alpha_bev = 0.6

    time_horizon = 50

    num_ave = 100
    ave_tab = np.arange(num_ave)

    advertised_type = 'hev'  # 'hev', 'phev', or 'bev'

    if network_type == "SL":
        L = 50  # linear system size (N = L x L: number of agents)
        network_parameters = (L,)
        folder_name = f'{network_type}_{L}L_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp_{alpha_phev}aphev_{alpha_bev}abev_{advertised_type}'
    elif network_type == "WS":
        N = 2500  # number of agents
        k = 4  # average node degree (must be divisible by 2)
        beta = 1  # rewiring probability
        network_parameters = (N, k, beta)
        folder_name = f'{network_type}_{N}N_{k}k_{beta}beta_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp_{alpha_phev}aphev_{alpha_bev}abev_{advertised_type}'

    num_h = 20
    h_start = 0
    h_end = 4
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
                             heterogeneous_susceptibilities=heterogeneous_susceptibilities,
                             heterogeneous_driving_patterns=heterogeneous_driving_patterns,
                             alpha_phev=alpha_phev,
                             alpha_bev=alpha_bev,
                             advertised_type=advertised_type,
                             h=item,
                             time_horizon=time_horizon,
                             folder_name=folder_name),
                     ave_tab)
