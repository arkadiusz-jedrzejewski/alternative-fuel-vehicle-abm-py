import numpy as np
import os
import matplotlib.pyplot as plt
import time
import datetime

def error_measure(target, data):
    return np.mean((target - data) ** 2)


data_dir = r"D:\Data\alternative-fuel-vehicle-abm-data"
adoption_target = [0.489, 0.319, 0.192]
alpha_phevs = np.linspace(4, 15, 12)  # [2, 3, 4, 5]
alpha_bevs = np.linspace(0, 1.5, 16)  # [0.2, 0.3, 0.4, 0.5]

parm_space = [(alpha_phev, alpha_bev) for alpha_phev in alpha_phevs for alpha_bev in alpha_bevs]
num_parms = len(parm_space)

network_type = "SL"  # "SL" - square lattice, "WS" Watts Strogatz network

heterogeneous_susceptibilities = 0
heterogeneous_driving_patterns = 0

h_hev, h_phev, h_bev = 0, 0, 0

time_horizon = 20

num_ave = 500
ave_tab = np.arange(num_ave)

error_dict = {}
start = time.time()
counter = 0
for alpha_phev, alpha_bev in parm_space:
    if network_type == "SL":
        L = 32  # linear system size (N = L x L: number of agents)
        network_parameters = (L,)
        model_name = f"{network_type}_{L}L_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp"
        folder_name = data_dir + f"/{model_name}" + f"_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}"
    elif network_type == "WS":
        N = 1024  # number of agents
        k = 4  # average node degree (must be divisible by 2)
        beta = 0  # rewiring probability
        network_parameters = (N, k, beta)
        model_name = f"{network_type}_{N}N_{k}k_{beta}beta_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp"
        folder_name = data_dir + f"/{model_name}" + f"_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}"

    total_error = 0
    for num_ave in ave_tab:
        file_name = folder_name + f"/sim-{num_ave}.txt"
        # print(file_name)
        adoption_sim = np.loadtxt(file_name)[-1, :3]
        # print(adoption_sim)
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

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(alpha_phevs_m, alpha_bevs_m, errors, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
plt.show()

folder_name = data_dir + "/map_results"
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

np.savetxt(folder_name + f"/{model_name}_mses.csv", errors, fmt="%.18f")
np.savetxt(folder_name + f"/{model_name}_alpha_phevs_m.csv", alpha_phevs_m, fmt="%.18f")
np.savetxt(folder_name + f"/{model_name}_alpha_bevs_m.csv", alpha_bevs_m, fmt="%.18f")

best_parms = (best_parms[0], best_parms[1], error_dict[best_parms])
np.savetxt(folder_name + f"/{model_name}_best_parms.csv", best_parms, fmt="%.18f")
