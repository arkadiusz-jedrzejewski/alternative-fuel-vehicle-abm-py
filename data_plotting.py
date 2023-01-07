import numpy as np
import matplotlib.pyplot as plt

folder_name = 'SL_50L_1hs_1hdp_8aphev_0.6abev_hev'

ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')


def get_mean_and_quantiles(data):
    q95 = np.quantile(data, q=0.95, axis=0)
    q05 = np.quantile(data, q=0.05, axis=0)
    mean = np.mean(data, axis=0)
    return mean, q05, q95


def get_trajectory(number):
    traj = {
        'hev': np.zeros([ave_num, 51]),
        'phev': np.zeros([ave_num, 51]),
        'bev': np.zeros([ave_num, 51]),
        'none': np.zeros([ave_num, 51])
    }

    for i in range(ave_num):
        file_name = f'\\{number}\\sim-{i}.txt'
        path = folder_name + '\\' + file_name
        for j, item in enumerate(traj.keys()):
            traj[item][i, :] = np.loadtxt(path)[:, j]

    return traj


num = 20
means = {
    'hev': np.zeros([1, num]),
    'phev': np.zeros([1, num]),
    'bev': np.zeros([1, num]),
    'none': np.zeros([1, num])
}

for i in range(num):
    traj = get_trajectory(i)
    for vehicle_type in traj.keys():
        mean, q05, q95 = get_mean_and_quantiles(traj[vehicle_type])
        means[vehicle_type][0, i] = mean[-1]


def plot_fig(fig, ax, flag=True):
    num = 20
    means = {
        'hev': np.zeros([1, num]),
        'phev': np.zeros([1, num]),
        'bev': np.zeros([1, num]),
        'none': np.zeros([1, num])
    }
    for i in range(num):
        traj = get_trajectory(i)
        for vehicle_type in traj.keys():
            mean, q05, q95 = get_mean_and_quantiles(traj[vehicle_type])
            means[vehicle_type][0, i] = mean[-1]

    for vehicle_type in traj.keys():
        if flag:
            ax.plot(h_values[:, 1], means[vehicle_type][0, :], '.--')
        else:
            ax.plot(h_values[:, 1], means[vehicle_type][0, :], 'x--')
        ax.set_ylim([0, 1])


fig, axs = plt.subplots(1, 3)

folder_name = 'SL_50L_1hs_1hdp_8aphev_0.6abev_hev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[0])

folder_name = 'SL_50L_1hs_1hdp_8aphev_0.6abev_phev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[1])

folder_name = 'SL_50L_1hs_1hdp_8aphev_0.6abev_bev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[2])

folder_name = 'SL_50L_0hs_0hdp_8aphev_0.6abev_hev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[0], False)

folder_name = 'SL_50L_0hs_0hdp_8aphev_0.6abev_phev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[1], False)

folder_name = 'SL_50L_0hs_0hdp_8aphev_0.6abev_bev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[2], False)

plt.show()

fig, axs = plt.subplots(1, 3)

folder_name = 'SL_50L_1hs_1hdp_8aphev_0.6abev_hev'
folder_name = 'WS_2500N_4k_0beta_1hs_1hdp_8aphev_0.6abev_hev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[0])

folder_name = 'SL_50L_1hs_1hdp_8aphev_0.6abev_phev'
folder_name = 'WS_2500N_4k_0beta_1hs_1hdp_8aphev_0.6abev_phev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[1])

folder_name = 'SL_50L_1hs_1hdp_8aphev_0.6abev_bev'
folder_name = 'WS_2500N_4k_0beta_1hs_1hdp_8aphev_0.6abev_bev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[2])

folder_name = 'WS_2500N_4k_1beta_1hs_1hdp_8aphev_0.6abev_hev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[0], False)

folder_name = 'WS_2500N_4k_1beta_1hs_1hdp_8aphev_0.6abev_phev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[1], False)

folder_name = 'WS_2500N_4k_1beta_1hs_1hdp_8aphev_0.6abev_bev'
ave_num = int(np.loadtxt(folder_name + '\\num_ave.csv'))
h_values = np.loadtxt(folder_name + '\\h_values.csv')
plot_fig(fig, axs[2], False)

plt.show()
