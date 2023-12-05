import os
import numpy as np
import subprocess
import multiprocessing as mp
from functools import partial
import time
import datetime
import matplotlib.pyplot as plt
import tikzplotlib


def create_file(sim_num,
                network_params,
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
    network_type = network_params[0]
    if network_type == "SL":
        (_, L) = network_params
        subprocess.run(
            f"alternative_fuel_vehicle_abm.exe {network_type} {heterogeneous_hev_susceptibilities} {heterogeneous_phev_susceptibilities} {heterogeneous_bev_susceptibilities} {heterogeneous_driving_patterns} {alpha_phev} {alpha_bev} {h_hev} {h_phev} {h_bev} {time_horizon} {file_name} {L}")
    elif network_type == "WS":
        (_, N, k, beta) = network_params
        subprocess.run(
            f"alternative_fuel_vehicle_abm.exe {network_type} {heterogeneous_hev_susceptibilities} {heterogeneous_phev_susceptibilities} {heterogeneous_bev_susceptibilities} {heterogeneous_driving_patterns} {alpha_phev} {alpha_bev} {h_hev} {h_phev} {h_bev} {time_horizon} {file_name} {N} {k} {beta}")
        return 0


def error_measure(target, data):
    return np.mean((target - data) ** 2)


def get_network_name(network_params):
    network_type = network_params[0]
    if network_type == "SL":
        (_, L) = network_params  # linear system size (N = L x L: number of agents)
        network_name = f"{network_type}_{L}L"
    elif network_type == "WS":
        (_, N, k, beta) = network_params
        network_name = f"{network_type}_{N}N_{k}k_{beta}beta"
    return network_name


def get_model_name(network_name, heterogeneous_hev_susceptibilities, heterogeneous_phev_susceptibilities,
                   heterogeneous_bev_susceptibilities, heterogeneous_driving_patterns):
    return f"{network_name}_{heterogeneous_hev_susceptibilities}_{heterogeneous_phev_susceptibilities}_{heterogeneous_bev_susceptibilities}hs_{heterogeneous_driving_patterns}hdp"


def get_folder_name(model_name, alpha_phev, alpha_bev, hs, data_dir):
    h_hev, h_phev, h_bev = hs
    folder_name = data_dir + f"/{model_name}" + f"_{alpha_phev}aphev_{alpha_bev}abev_{h_hev}_{h_phev}_{h_bev}"
    return folder_name


def run_single_simulation(network_params, heterogeneous_hev_susceptibilities, heterogeneous_phev_susceptibilities,
                          heterogeneous_bev_susceptibilities, heterogeneous_driving_patterns,
                          alpha_phev,
                          alpha_bev, h_hev, h_phev, h_bev, num_ave, time_horizon, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    np.savetxt(folder_name + '/num_ave.csv', [num_ave], fmt="%i")
    np.savetxt(folder_name + '/model_params.csv', np.column_stack([heterogeneous_hev_susceptibilities,
                                                                   heterogeneous_phev_susceptibilities,
                                                                   heterogeneous_bev_susceptibilities,
                                                                   heterogeneous_driving_patterns,
                                                                   alpha_phev,
                                                                   alpha_bev,
                                                                   h_hev,
                                                                   h_phev,
                                                                   h_bev,
                                                                   time_horizon]),
               fmt=("%i," * 4 + "%.18f," * 5 + "%i"))

    sim_numbers = np.arange(num_ave)
    with mp.Pool(6) as pool:
        pool.map(partial(create_file,
                         network_params=network_params,
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


def run_automated_calibration(data_dir, alpha_phevs, alpha_bevs, network_params, heterogeneous_hev_susceptibilities,
                              heterogeneous_phev_susceptibilities, heterogeneous_bev_susceptibilities,
                              heterogeneous_driving_patterns):
    adoption_target = [0.489, 0.319, 0.192]

    num_parms = len(alpha_phevs) * len(alpha_bevs)

    parm_space = [(alpha_phev, alpha_bev) for alpha_phev in alpha_phevs for alpha_bev in alpha_bevs]

    model_name = get_model_name(network_name=get_network_name(network_params),
                                heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                                heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                                heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                                heterogeneous_driving_patterns=heterogeneous_driving_patterns)

    start = time.time()
    counter = 0

    error_dict = {}
    for alpha_phev, alpha_bev in parm_space:
        folder_name = get_folder_name(model_name=model_name,
                                      alpha_phev=alpha_phev,
                                      alpha_bev=alpha_bev,
                                      hs=(0, 0, 0),
                                      data_dir=data_dir)
        run_single_simulation(network_params=network_params,
                              heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                              heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                              heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                              heterogeneous_driving_patterns=heterogeneous_driving_patterns,
                              alpha_phev=alpha_phev,
                              alpha_bev=alpha_bev,
                              h_hev=0,
                              h_phev=0,
                              h_bev=0,
                              num_ave=500,
                              time_horizon=20,
                              folder_name=folder_name)
        sim_numbers = np.arange(int(np.loadtxt(folder_name + "/num_ave.csv")))
        total_error = 0
        for sim_number in sim_numbers:
            file_name = folder_name + f"/sim-{sim_number}.txt"
            adoption_sim = np.loadtxt(file_name)[-1, :3]
            error = error_measure(adoption_target, adoption_sim)
            total_error += error
        total_error /= len(sim_numbers)
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


def run_diagram_simulations(cal_data_dir, data_dir, network_params, heterogeneous_hev_susceptibilities,
                            heterogeneous_phev_susceptibilities, heterogeneous_bev_susceptibilities,
                            heterogeneous_driving_patterns):
    model_name = get_model_name(network_name=get_network_name(network_params),
                                heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                                heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                                heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                                heterogeneous_driving_patterns=heterogeneous_driving_patterns)
    alpha_phev, alpha_bev, mse = np.loadtxt(cal_data_dir + f"/map_results/{model_name}_best_parms.csv")

    print(
        f"{network_params} {heterogeneous_hev_susceptibilities}_{heterogeneous_phev_susceptibilities}_{heterogeneous_bev_susceptibilities}hs_{heterogeneous_driving_patterns}hdp:\ta_phev={alpha_phev}\ta_bev={alpha_bev}\tMSE={mse}")

    num_h = 20
    h_start = 0
    h_end = 4
    hs = np.linspace(h_start, h_end, num_h)
    h_inds = np.arange(num_h)

    h_inds_tuples = [(h, 0, 0) for h in h_inds] + [(0, h, 0) for h in h_inds[1:]] + [(0, 0, h) for h in h_inds[1:]]
    h_tuples = [(h, 0, 0) for h in hs] + [(0, h, 0) for h in hs[1:]] + [(0, 0, h) for h in hs[1:]]

    data_dir = data_dir + "/" + model_name
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    np.savetxt(data_dir + '/h_values.csv', np.column_stack((h_inds, hs)), fmt=("%i", "%.18f"))

    start = time.time()
    counter = 0
    for ind, h_inds in enumerate(h_inds_tuples):
        (h_hev, h_phev, h_bev) = h_tuples[ind]
        folder_name = get_folder_name(model_name=model_name,
                                      alpha_phev=alpha_phev,
                                      alpha_bev=alpha_bev,
                                      hs=h_inds,
                                      data_dir=data_dir)

        run_single_simulation(network_params=network_params,
                              heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                              heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                              heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                              heterogeneous_driving_patterns=heterogeneous_driving_patterns,
                              alpha_phev=alpha_phev,
                              alpha_bev=alpha_bev,
                              h_hev=h_hev,
                              h_phev=h_phev,
                              h_bev=h_bev,
                              num_ave=500,
                              time_horizon=20,
                              folder_name=folder_name)
        end = time.time()
        counter += 1
        ave_time = (end - start) / counter
        print(f"{h_inds}\testimated remaining time:\t" + str(
            datetime.timedelta(seconds=int((len(h_inds_tuples) - counter) * ave_time))))


def get_trajectories(folder_name):
    num_ave = int(np.loadtxt(folder_name + "/num_ave.csv"))
    time_horizon = int(np.loadtxt(folder_name + "/model_params.csv", delimiter=",")[-1])
    time_horizon += 1
    trajectories = {
        'hev': np.zeros([num_ave, time_horizon]),
        'phev': np.zeros([num_ave, time_horizon]),
        'bev': np.zeros([num_ave, time_horizon]),
        'none': np.zeros([num_ave, time_horizon])
    }

    for i in range(num_ave):
        for j, item in enumerate(trajectories.keys()):
            trajectories[item][i, :] = np.loadtxt(folder_name + f'/sim-{i}.txt')[:, j]

    return trajectories


def calculate_mean_and_quantiles(data):
    q95 = np.quantile(data, q=0.95, axis=0)
    q05 = np.quantile(data, q=0.05, axis=0)
    mean = np.mean(data, axis=0)
    sem = np.std(data, axis=0, ddof=1) / np.sqrt(np.size(data, axis=0)) # standard error of the mean
    return mean, q05, q95, sem


def get_mean_and_quantiles(folder_name):
    trajectories = get_trajectories(folder_name=folder_name)
    result = []
    for vehicle_type in trajectories.keys():
        results = calculate_mean_and_quantiles(trajectories[vehicle_type])
        result.append(results)
    # print(result)
    return result


def get_diagram_data(cal_data_dir, data_dir, network_params, heterogeneous_hev_susceptibilities,
                     heterogeneous_phev_susceptibilities, heterogeneous_bev_susceptibilities,
                     heterogeneous_driving_patterns, h_type):
    model_name = get_model_name(network_name=get_network_name(network_params),
                                heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                                heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                                heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                                heterogeneous_driving_patterns=heterogeneous_driving_patterns)
    alpha_phev, alpha_bev, mse = np.loadtxt(cal_data_dir + f"/map_results/{model_name}_best_parms.csv")
    data_dir_path = data_dir + "/" + model_name
    h_values = np.loadtxt(data_dir_path + '/h_values.csv')

    hevs, phevs, bevs, nones, = [], [], [], []
    sem_hevs, sem_phevs, sem_bevs, sem_nones, = [], [], [], []
    for h_index, h in h_values:
        h_index = int(h_index)

        match h_type:
            case "h_hev":
                hs = (h_index, 0, 0)
            case "h_phev":
                hs = (0, h_index, 0)
            case "h_bev":
                hs = (0, 0, h_index)
        folder_name = get_folder_name(model_name=model_name,
                                      alpha_phev=alpha_phev,
                                      alpha_bev=alpha_bev,
                                      hs=hs,
                                      data_dir=data_dir_path)
        hev, phev, bev, none = get_mean_and_quantiles(folder_name)
        hevs.append(hev[0][-1])
        sem_hevs.append(hev[3][-1])
        phevs.append(phev[0][-1])
        sem_phevs.append(phev[3][-1])
        bevs.append(bev[0][-1])
        sem_bevs.append(bev[3][-1])
        nones.append(none[0][-1])
        sem_nones.append(none[3][-1])

    result_folder_name = data_dir + f"/diagram_data"
    if not os.path.exists(result_folder_name):
        os.mkdir(result_folder_name)

    data = (h_values[:, 1], hevs, sem_hevs, phevs, sem_phevs, bevs, sem_bevs, nones, sem_nones)
    np.savetxt(result_folder_name + "/" + model_name + "_" + h_type + ".csv", np.transpose(data), delimiter=',')
    return h_values[:, 1], hevs, sem_hevs, phevs, sem_phevs, bevs, sem_bevs, nones, sem_nones


def get_calibration_map(cal_data_dir, network_params, heterogeneous_hev_susceptibilities,
                        heterogeneous_phev_susceptibilities, heterogeneous_bev_susceptibilities,
                        heterogeneous_driving_patterns):
    model_name = get_model_name(network_name=get_network_name(network_params),
                                heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                                heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                                heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                                heterogeneous_driving_patterns=heterogeneous_driving_patterns)
    folder_name = cal_data_dir + "/map_results"
    errors = np.loadtxt(folder_name + f"/{model_name}_mses.csv", delimiter=",")
    alpha_phevs_m = np.loadtxt(folder_name + f"/{model_name}_alpha_phevs_m.csv", delimiter=",")
    alpha_bevs_m = np.loadtxt(folder_name + f"/{model_name}_alpha_bevs_m.csv", delimiter=",")

    # best_parms = (best_parms[0], best_parms[1], error_dict[best_parms])
    alpha_phev, alpha_bev, mse = np.loadtxt(folder_name + f"/{model_name}_best_parms.csv")
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(alpha_phevs_m, alpha_bevs_m, errors, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none', alpha=0.9)
    ax.scatter(alpha_phev, alpha_bev, mse, c='r')
    ax.set_xlabel(r'alpha_PHEV$')
    ax.set_ylabel(r'$alpha_BEV$')
    ax.set_zlabel("MSE")
    plt.title(f"{model_name} aphev={alpha_phev:.0f} abev={alpha_bev:.1f} MSE={mse:.5f}")

    title_string = f"{model_name} aphev={alpha_phev:.0f} abev={alpha_bev:.1f}"
    plt.savefig(folder_name + "/" + title_string + ".png")
    tikzplotlib.save(folder_name + "/" + title_string + ".tex")


def get_calibration(cal_data_dir, network_params, heterogeneous_hev_susceptibilities,
                    heterogeneous_phev_susceptibilities, heterogeneous_bev_susceptibilities,
                    heterogeneous_driving_patterns):
    model_name = get_model_name(network_name=get_network_name(network_params),
                                heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                                heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                                heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                                heterogeneous_driving_patterns=heterogeneous_driving_patterns)
    alpha_phev, alpha_bev, mse = np.loadtxt(cal_data_dir + f"/map_results/{model_name}_best_parms.csv")
    folder_name = cal_data_dir + "/" + model_name + f"_{alpha_phev}aphev_{alpha_bev}abev_0_0_0"
    result = get_mean_and_quantiles(folder_name)

    adoption_target = [0.489, 0.319, 0.192]
    for i in range(len(adoption_target)):
        plt.plot([0, len(result[0][0]) - 1], [adoption_target[i]] * 2)

    plt.gca().set_prop_cycle(None)  # reset color cycle

    labels = ["HEV", "PHEV", "BEV", "NONE"]
    for i in range(4):
        plt.fill_between(range(len(result[i][1])), result[i][1], result[i][2], alpha=0.2)
        # plt.plot(range(len(result[i][0])), result[i][0],':.')
        plt.errorbar(range(len(result[i][0])), result[i][0], yerr=result[i][3], label=labels[i])

    title_string = f"{model_name} aphev={alpha_phev:.0f} abev={alpha_bev:.1f}"
    plt.title(title_string)
    plt.xlabel("MCS")
    plt.ylabel("Adoption level")
    plt.xlim([0, len(result[0][0]) - 1])
    plt.ylim([0, 1])
    # plt.legend(loc="upper right")

    result_folder_name = cal_data_dir + f"/calibration_plots"
    if not os.path.exists(result_folder_name):
        os.mkdir(result_folder_name)

    plt.savefig(result_folder_name + "/" + title_string + ".png")
    tikzplotlib.save(result_folder_name + "/" + title_string + ".tex")
    plt.clf()


def get_calibration_ax(ax, cal_data_dir, network_params, heterogeneous_hev_susceptibilities,
                       heterogeneous_phev_susceptibilities, heterogeneous_bev_susceptibilities,
                       heterogeneous_driving_patterns):
    model_name = get_model_name(network_name=get_network_name(network_params),
                                heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                                heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                                heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                                heterogeneous_driving_patterns=heterogeneous_driving_patterns)
    alpha_phev, alpha_bev, mse = np.loadtxt(cal_data_dir + f"/map_results/{model_name}_best_parms.csv")
    folder_name = cal_data_dir + "/" + model_name + f"_{alpha_phev}aphev_{alpha_bev}abev_0_0_0"
    result = get_mean_and_quantiles(folder_name)

    adoption_target = [0.489, 0.319, 0.192]
    for i in range(len(adoption_target)):
        ax.plot([0, len(result[0][0]) - 1], [adoption_target[i]] * 2)

    ax.set_prop_cycle(None)  # reset color cycle

    labels = ["HEV", "PHEV", "BEV", "NONE"]
    for i in range(4):
        ax.fill_between(range(len(result[i][1])), result[i][1], result[i][2], alpha=0.2)
        ax.errorbar(range(len(result[i][0])), result[i][0], yerr=result[i][3], fmt='.')
        # ax.plot(range(len(result[i][0])), result[i][0], '.')

    # title_string = f"{model_name} aphev={alpha_phev:.0f} abev={alpha_bev:.1f}"
    # ax.set_title(title_string)
    ax.set_xlabel("MCS")
    ax.set_ylabel("Adoption level")
    ax.set_xlim([0, len(result[0][0]) - 1])
    ax.set_ylim([0, 1])
    # plt.legend(loc="upper right")

    result_folder_name = cal_data_dir + f"/calibration_plots"
    # if not os.path.exists(result_folder_name):
    #     os.mkdir(result_folder_name)
    return result_folder_name + "/" + get_network_name(network_params) + ".tex"
    # plt.savefig(result_folder_name + "/" + title_string + ".png")
    # tikzplotlib.save(result_folder_name + "/" + title_string + ".tex")
    # plt.clf()


def get_diagram(cal_data_dir, data_dir, network_params, h_type):
    result_folder_name = data_dir + f"/diagram_data"
    network_name = get_network_name(network_params=network_params)
    title_string = ""
    mark_it = 0
    marks = ["d", ".", "^", "x"]
    car_type = ["HEV", "PHEV", "BEV"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 6))
    for heterogeneous_hev_susceptibilities in [0, 1]:
        heterogeneous_phev_susceptibilities = heterogeneous_hev_susceptibilities
        heterogeneous_bev_susceptibilities = heterogeneous_hev_susceptibilities
        # for heterogeneous_phev_susceptibilities in [0, 1]:
        #     for heterogeneous_bev_susceptibilities in [0, 1]:
        for heterogeneous_driving_patterns in [0, 1]:
            model_name = get_model_name(network_name=network_name,
                                        heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                                        heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                                        heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                                        heterogeneous_driving_patterns=heterogeneous_driving_patterns)
            alpha_phev, alpha_bev, mse = np.loadtxt(cal_data_dir + f"/map_results/{model_name}_best_parms.csv")
            title_string += f"{model_name} aphev={alpha_phev:.0f} abev={alpha_bev:.1f} {h_type}\n"
            data = np.loadtxt(result_folder_name + "/" + model_name + "_" + h_type + ".csv", delimiter=',')
            mark_t = 3
            if h_type == "h_hev" and heterogeneous_hev_susceptibilities == 1:
                mark_t = 0
            elif h_type == "h_phev" and heterogeneous_phev_susceptibilities == 1:
                mark_t = 0
            elif h_type == "h_bev" and heterogeneous_bev_susceptibilities == 1:
                mark_t = 0
            for i in range(3):
                ax[i].errorbar(data[:, 0],
                               data[:, 1 + i * 2],
                               yerr=data[:, 2 + i * 2],
                               fmt=":" + marks[mark_t],
                               markersize=5,
                               label=f"{heterogeneous_hev_susceptibilities}_{heterogeneous_phev_susceptibilities}_{heterogeneous_bev_susceptibilities}hs_{heterogeneous_driving_patterns}hdp")
                # ax[i].plot(data[:, 0],
                #            data[:, 1 + i * 2],
                #            markersize=5)
                ax[i].set_ylim([0, 1])
                ax[i].set_xlabel(h_type)
                # ax[i].set_title(car_type[i])
                ax[i].set_ylabel("Adoption level")
                mark_it += 1
    # plt.suptitle(title_string)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plot_folder_name = data_dir + f"/diagram_plots"
    if not os.path.exists(plot_folder_name):
        os.mkdir(plot_folder_name)
    print(plot_folder_name + "/" + network_name + f"_{h_type}" + ".tex")
    # tikzplotlib.save(plot_folder_name + "/" + network_name + f"_{h_type}" + ".tex", axis_height='\\figH',
    #                  axis_width='\\figW')
    # plt.savefig(plot_folder_name + "/" + network_name + f"_{h_type}" + ".png")
    plt.show()
