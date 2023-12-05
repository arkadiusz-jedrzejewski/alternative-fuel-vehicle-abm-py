import os
import numpy as np
import subprocess
import multiprocessing as mp
from functools import partial


def create_file(sim_num,
                network_type,
                network_parameters,
                heterogeneous_hev_susceptibilities,
                heterogeneous_phev_susceptibilities,
                heterogeneous_bev_susceptibilities,
                heterogeneous_driving_patterns,
                alpha_phev,
                alpha_bev,
                h_hev,
                h_phev,
                h_bev,
                time_horizon,
                folder_name):
    file_name = folder_name + f"/sim-{sim_num}.txt"
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

    network_type = "SL"  # "SL" - square lattice, "WS" Watts Strogatz network

    heterogeneous_hev_susceptibilities = 1
    heterogeneous_phev_susceptibilities = 1
    heterogeneous_bev_susceptibilities = 1
    heterogeneous_driving_patterns = 0

    alpha_phev = 12  #15  #
    alpha_bev = 0.4  #4.5  #

    h_hev = 0
    h_phev = 0
    h_bev = 0

    time_horizon = 50

    num_ave = 100
    sim_numbers = np.arange(num_ave)

    if network_type == "SL":
        L = 50  # linear system size (N = L x L: number of agents)
        network_parameters = (L,)
        folder_name = f'{network_type}_{L}L_{heterogeneous_hev_susceptibilities}_{heterogeneous_phev_susceptibilities}_{heterogeneous_bev_susceptibilities}hs_{heterogeneous_driving_patterns}hdp_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}'
    elif network_type == "WS":
        N = 100  # number of agents
        k = 4  # average node degree (must be divisible by 2)
        beta = 0  # rewiring probability
        network_parameters = (N, k, beta)
        folder_name = f'{network_type}_{N}N_{k}k_{beta}beta_{heterogeneous_hev_susceptibilities}_{heterogeneous_phev_susceptibilities}_{heterogeneous_bev_susceptibilities}hs_{heterogeneous_driving_patterns}hdp_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}'

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    np.savetxt(folder_name + '/num_ave.csv', [num_ave], fmt="%i")

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
                         h_hev=h_hev,
                         h_phev=h_phev,
                         h_bev=h_bev,
                         time_horizon=time_horizon,
                         folder_name=folder_name),
                 sim_numbers)
