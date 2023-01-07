import os
import numpy as np
import subprocess
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import time
import datetime


def create_file(num_ave,
                network_params,
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
    network_type = network_params[0]
    if network_type == "SL":
        (_, L) = network_params
        subprocess.run(
            f"alternative_fuel_vehicle_abm.exe {network_type} {heterogeneous_susceptibilities} {heterogeneous_driving_patterns} {alpha_phev} {alpha_bev} {h_hev} {h_phev} {h_bev} {time_horizon} {file_name} {L}")
    elif network_type == "WS":
        (_, N, k, beta) = network_params
        subprocess.run(
            f"alternative_fuel_vehicle_abm.exe {network_type} {heterogeneous_susceptibilities} {heterogeneous_driving_patterns} {alpha_phev} {alpha_bev} {h_hev} {h_phev} {h_bev} {time_horizon} {file_name} {N} {k} {beta}")
        return 0


def error_measure(target, data):
    return np.mean((target - data) ** 2)

def run_automated_calibration(data_dir, alpha_phevs, alpha_bevs, network_params, heterogeneous_susceptibilities, heterogeneous_driving_patterns):
    adoption_target = [0.489, 0.319, 0.192]

    num_parms = len(alpha_phevs) * len(alpha_bevs)

    parm_space = [(alpha_phev, alpha_bev) for alpha_phev in alpha_phevs for alpha_bev in alpha_bevs]

    network_type = network_params[0]  # "SL" - square lattice, "WS" Watts Strogatz network

    h_hev, h_phev, h_bev = 0, 0, 0

    time_horizon = 20

    num_ave = 500
    ave_tab = np.arange(num_ave)

    start = time.time()
    counter = 0

    error_dict = {}
    for alpha_phev, alpha_bev in parm_space:
        if network_type == "SL":
            (_, L) = network_params  # linear system size (N = L x L: number of agents)
            model_name = f"{network_type}_{L}L_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp"
            folder_name = data_dir + f"/{model_name}" + f"_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}"
        elif network_type == "WS":
            (_, N, k, beta) = network_params
            model_name = f"{network_type}_{N}N_{k}k_{beta}beta_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp"
            folder_name = data_dir + f"/{model_name}" + f"_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}"

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        np.savetxt(folder_name + '/num_ave.csv', [num_ave], fmt="%i")

        with mp.Pool(6) as pool:
            pool.map(partial(create_file,
                             network_params=network_params,
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
            adoption_sim = np.loadtxt(file_name)[-1, :3]
            error = error_measure(adoption_target, adoption_sim)
            total_error += error
        total_error /= len(ave_tab)
        end = time.time()
        counter += 1
        ave_time = (end - start) / counter
        print(f"(alpha_phev, alpha_bev):\t({alpha_phev},{alpha_bev})\tMSE:\t{total_error}")
        print("estimated remaining time:\t" + str(datetime.timedelta(seconds=int((num_parms - counter) * ave_time))))
        error_dict[alpha_phev, alpha_bev] = total_error

    best_parms = min(error_dict, key=error_dict.get)
    print("Best parameters:", best_parms, "MSE:", error_dict[best_parms])

    alpha_phevs_m, alpha_bevs_m = np.meshgrid(alpha_phevs, alpha_bevs)
    errors = np.zeros(alpha_phevs_m.shape)

    for in1, i1 in enumerate(alpha_phevs):
        for in2, i2 in enumerate(alpha_bevs):
            errors[in2, in1] = error_dict[(i1, i2)]

    folder_name = data_dir + "/map_results"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    np.savetxt(folder_name + f"/{model_name}_mses.csv", errors, fmt="%.18f", delimiter=", ")
    np.savetxt(folder_name + f"/{model_name}_alpha_phevs_m.csv", alpha_phevs_m, fmt="%.18f", delimiter=",")
    np.savetxt(folder_name + f"/{model_name}_alpha_bevs_m.csv", alpha_bevs_m, fmt="%.18f", delimiter=",")

    best_parms = (best_parms[0], best_parms[1], error_dict[best_parms])
    np.savetxt(folder_name + f"/{model_name}_best_parms.csv", best_parms, fmt="%.18f", delimiter=",")

if __name__ == '__main__':

    data_dir = r"D:\Data\alternative-fuel-vehicle-abm-data"
    alpha_phevs = np.linspace(4, 15, 12)
    alpha_bevs = np.linspace(0, 1.5, 16)
    networks = [("SL", 32),
                ("WS", 1024, 4, 1),
                ("WS", 1024, 4, 0)]

    for network_params in networks:
        for heterogeneous_susceptibilities in [0, 1]:
            for heterogeneous_driving_patterns in [0, 1]:
                run_automated_calibration(data_dir=data_dir,
                                          alpha_phevs=alpha_phevs,
                                          alpha_bevs=alpha_bevs,
                                          network_params=network_params,
                                          heterogeneous_susceptibilities=heterogeneous_susceptibilities,
                                          heterogeneous_driving_patterns=heterogeneous_driving_patterns)
