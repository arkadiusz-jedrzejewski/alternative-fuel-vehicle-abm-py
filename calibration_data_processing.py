import numpy as np
import matplotlib.pyplot as plt


def get_mean_and_quantiles(data):
    q95 = np.quantile(data, q=0.95, axis=0)
    q05 = np.quantile(data, q=0.05, axis=0)
    mean = np.mean(data, axis=0)
    sem = np.std(data, ddof=1, axis=0) / np.sqrt(data.shape[0])  # standard error of the mean
    return mean, q05, q95, sem


def get_trajectory(number):
    traj = {
        'hev': np.zeros([ave_num, 21]),
        'phev': np.zeros([ave_num, 21]),
        'bev': np.zeros([ave_num, 21]),
        'none': np.zeros([ave_num, 21])
    }

    for i in range(ave_num):
        file_name = f'\\sim-{i}.txt'
        path = folder_name + '\\' + file_name
        for j, item in enumerate(traj.keys()):
            traj[item][i, :] = np.loadtxt(path)[:, j]

    return traj


data_dir = r"D:\Data\alternative-fuel-vehicle-abm-hypcal-data-3"

network_type = "WS"
heterogeneous_susceptibilities = 1
heterogeneous_driving_patterns = 1
#h_hev, h_phev, h_bev = 0, 0, 1

# if network_type == "SL":
#     L = 32  # linear system size (N = L x L: number of agents)
#     network_parameters = (L,)
#     model_name = f"{network_type}_{L}L_{heterogeneous_susceptibilities}_{heterogeneous_susceptibilities}_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp"
#     best_parms = np.loadtxt(f"{data_dir}/map_results/{model_name}_best_parms.csv")
#     alpha_phev, alpha_bev, mse = best_parms[0], best_parms[1],  best_parms[2]
#     print(alpha_phev, alpha_bev, mse)
#     print(best_parms)
#     folder_name = data_dir + f"/{model_name}" + f"_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}"
# elif network_type == "WS":
#     N = 1024  # number of agents
#     k = 4  # average node degree (must be divisible by 2)
#     beta = 1  # rewiring probability
#     network_parameters = (N, k, beta)
#     model_name = f"{network_type}_{N}N_{k}k_{beta}beta_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp"
#     best_parms = np.loadtxt(f"{data_dir}/map_results/{model_name}_best_parms.csv")
#     alpha_phev, alpha_bev, mse = best_parms[0], best_parms[1], best_parms[2]
#     print(alpha_phev, alpha_bev, mse)
#     print(best_parms)
#
#     folder_name = data_dir + f"/{model_name}" + f"_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}"
if network_type == "SL":
    L = 32  # linear system size (N = L x L: number of agents)
    network_parameters = (L,)
    model_name = f"{network_type}_{L}L_{heterogeneous_susceptibilities}_{heterogeneous_susceptibilities}_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp"
    best_parms = np.loadtxt(f"{data_dir}/{model_name}_best_parms.csv")
    alpha_phev, alpha_bev, h_hev, h_phev, h_bev, mse = best_parms[0], best_parms[1],  best_parms[2], best_parms[3], best_parms[4], best_parms[5]
    print(alpha_phev, alpha_bev, h_hev, h_phev, h_bev, mse)
    print(best_parms)
    folder_name = data_dir + f"/{model_name}" + f"_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}"
elif network_type == "WS":
    N = 1024  # number of agents
    k = 4  # average node degree (must be divisible by 2)
    beta = 0  # rewiring probability
    network_parameters = (N, k, beta)
    model_name = f"{network_type}_{N}N_{k}k_{beta}beta_{heterogeneous_susceptibilities}_{heterogeneous_susceptibilities}_{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp"
    best_parms = np.loadtxt(f"{data_dir}/{model_name}_best_parms.csv")
    alpha_phev, alpha_bev, mse = best_parms[0], best_parms[1], best_parms[2]
    alpha_phev, alpha_bev, h_hev, h_phev, h_bev, mse = best_parms[0], best_parms[1], best_parms[2], best_parms[3], \
    best_parms[4], best_parms[5]
    print(alpha_phev, alpha_bev, h_hev, h_phev, h_bev, mse)
    print(best_parms)

    folder_name = data_dir + f"/{model_name}" + f"_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}"

ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))

print(ave_num)

t_mcs = range(21)
traj = get_trajectory(0)

means = {
    'hev': np.zeros([1, 1]),
    'phev': np.zeros([1, 1]),
    'bev': np.zeros([1, 1]),
    'none': np.zeros([1, 1])
}
sems = {
    'hev': np.zeros([1, 1]),
    'phev': np.zeros([1, 1]),
    'bev': np.zeros([1, 1]),
    'none': np.zeros([1, 1])
}

handles = []
fig, ax = plt.subplots()

# calibration adoption levels
ax.plot([t_mcs[0], t_mcs[-1]], [0.489] * 2, 'b')
ax.plot([t_mcs[0], t_mcs[-1]], [0.319] * 2, 'y')
ax.plot([t_mcs[0], t_mcs[-1]], [0.192] * 2, 'g')

for vehicle_type in traj.keys():
    mean, q05, q95, sem = get_mean_and_quantiles(traj[vehicle_type])
    means[vehicle_type][0, 0] = mean[-1]
    sems[vehicle_type][0, 0] = sem[-1]
    ax.fill_between(t_mcs, q05, q95, alpha=0.2)
    handle, = ax.plot(t_mcs, mean, ':.')
    handles.append(handle)

ax.legend(handles, ['hev', 'phev', 'bev', 'none'])

print(means)
print(sems)

plt.title(model_name)
plt.xlabel("MCS")
plt.ylabel("adoption")
#plt.savefig(f'{folder_name}/{folder_name}.png')
plt.show()