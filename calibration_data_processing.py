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
        'hev': np.zeros([ave_num, 51]),
        'phev': np.zeros([ave_num, 51]),
        'bev': np.zeros([ave_num, 51]),
        'none': np.zeros([ave_num, 51])
    }

    for i in range(ave_num):
        file_name = f'\\sim-{i}.txt'
        path = folder_name + '\\' + file_name
        for j, item in enumerate(traj.keys()):
            traj[item][i, :] = np.loadtxt(path)[:, j]

    return traj


#folder_name = 'SL_50L_0hs_0hdp_8aphev_0.6abev_0_0_0'
folder_name = 'SL_50L_0hs_0hdp_12aphev_0.4abev_0_0_0'

#folder_name = 'SL_50L_1hs_1hdp_15aphev_4.5abev_0_0_0'


ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))

t_mcs = range(51)
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


plt.savefig(f'D:/OneDrive - Politechnika Wroclawska/manuscripts-epf/2021_cars/calibration/00/{folder_name}.png')
plt.show()