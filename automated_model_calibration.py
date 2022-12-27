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
                h_hev,
                h_phev,
                h_bev,
                time_horizon,
                folder_name):
    file_name = folder_name + f"/sim-{num_ave}.txt"
    if network_type == "SL":
        (L,) = network_parameters
        subprocess.run(
            f"alternative_fuel_vehicle_abm.exe {network_type} {heterogeneous_susceptibilities} {heterogeneous_driving_patterns} {alpha_phev} {alpha_bev} {h_hev} {h_phev} {h_bev} {time_horizon} {file_name} {L}")
    elif network_type == "WS":
        (N, k, beta) = network_parameters
        subprocess.run(
            f"alternative_fuel_vehicle_abm.exe {network_type} {heterogeneous_susceptibilities} {heterogeneous_driving_patterns} {alpha_phev} {alpha_bev} {h_hev} {h_phev} {h_bev} {time_horizon} {file_name} {N} {k} {beta}")
        return 0


def error_measure(target, data):
    return np.mean((target - data) ** 2)


if __name__ == '__main__':

    adoption_target = [0.489, 0.319, 0.192]

    alpha_phevs = [6, 7, 8]
    alpha_bevs = [0.4, 0.5, 0.6, 0.7]

    parm_space = [(alpha_phev, alpha_bev) for alpha_phev in alpha_phevs for alpha_bev in alpha_bevs]

    network_type = "SL"  # "SL" - square lattice, "WS" Watts Strogatz network

    heterogeneous_susceptibilities = 0
    heterogeneous_driving_patterns = 0

    h_hev, h_phev, h_bev = 0, 0, 0

    time_horizon = 50

    num_ave = 15
    ave_tab = np.arange(num_ave)

    error_dict = {}
    for alpha_phev, alpha_bev in parm_space:
        print(f"(alpha_phev, alpha_bev):\t({alpha_phev},{alpha_bev})")
        if network_type == "SL":
            L = 50  # linear system size (N = L x L: number of agents)
            network_parameters = (L,)
            folder_name = f'{network_type}_{L}L_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}'
        elif network_type == "WS":
            N = 100  # number of agents
            k = 4  # average node degree (must be divisible by 2)
            beta = 0  # rewiring probability
            network_parameters = (N, k, beta)
            folder_name = f'{network_type}_{N}N_{k}k_{beta}beta_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}'

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        np.savetxt(folder_name + '/num_ave.csv', [num_ave], fmt="%i")

        with mp.Pool(6) as pool:
            pool.map(partial(create_file,
                             network_type=network_type,
                             network_parameters=network_parameters,
                             heterogeneous_susceptibilities=heterogeneous_susceptibilities,
                             heterogeneous_driving_patterns=heterogeneous_driving_patterns,
                             alpha_phev=alpha_phev,
                             alpha_bev=alpha_bev,
                             h_hev=h_hev,
                             h_phev=h_phev,
                             h_bev=h_bev,
                             time_horizon=time_horizon,
                             folder_name=folder_name),
                     ave_tab)

        total_error = 0
        for num_ave in ave_tab:
            file_name = folder_name + f"/sim-{num_ave}.txt"
            #print(file_name)
            adoption_sim = np.loadtxt(file_name)[-1, :3]
            #print(adoption_sim)
            error = error_measure(adoption_target, adoption_sim)
            total_error += error
        total_error /= len(ave_tab)
        print("MSE:", total_error)
        error_dict[alpha_phev, alpha_bev] = total_error
        best_parms = min(error_dict, key=error_dict.get)

    print("Best parameters:", best_parms)
    print(len(ave_tab))
    print("MSE:", error_dict[best_parms])
    print(error_dict)
