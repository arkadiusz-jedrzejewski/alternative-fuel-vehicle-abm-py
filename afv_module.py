import os
import numpy as np
import subprocess
import multiprocessing as mp
from functools import partial
import time
import datetime
import matplotlib.pyplot as plt
import tikzplotlib
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials


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
    """
    runs a single simulation of the model
    """
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
    """
    runs parallely simulations with a given set of parameters
    :param network_params: tuple of network parameters
    :param heterogeneous_hev_susceptibilities: 1 if hev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_phev_susceptibilities: 1 if phev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_bev_susceptibilities: 1 if bev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_driving_patterns: 1 if driving patterns  are heterogeneous, 0 if homogeneous
    :param alpha_phev: phev calibration parameter that shows up in the formula for the refueling effect (RFE), see Eq.(5)
    :param alpha_bev: bev calibration parameter that shows up in the formula for the refueling effect (RFE), see Eq.(5)
    :param h_hev: strength of marketing for hev
    :param h_phev: strength of marketing for phev
    :param h_bev: strength of marketing for bev
    :param num_ave: number of independent simulations
    :param time_horizon: time length of the simulation in Monte Carlo steps (MCS)
    :param folder_name: name of the folder where the output is saved
    :return:
    """
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
    with mp.Pool(8) as pool:
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
                              heterogeneous_driving_patterns, hs_cal):
    """
    run the calibration based on the simple grid-search method, see Appendix B. Grid-search calibration
    for a chosen values of calibration parameters alpha_phev and alpha_bev stored in lists alpha_phevs and alpha_bevs,
    it runs simulations and calculates the means square error (MSE)
    Next, it finds the calibration parameters that minimize MSE
    :param data_dir: path to the folder where the calibration data is saved
    :param alpha_phevs: list of phev calibration parameters that show up in the formula for the refueling effect (RFE), see Eq.(5)
    :param alpha_bevs: list of bev calibration parameters that shows up in the formula for the refueling effect (RFE), see Eq.(5)
    :param network_params: tuple of network parameters
    :param heterogeneous_hev_susceptibilities: 1 if hev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_phev_susceptibilities: 1 if phev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_bev_susceptibilities: 1 if bev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_driving_patterns: 1 if driving patterns are heterogeneous, 0 if homogeneous
    :param hs_cal: tuple of marketing strengths (h_hev, h_phev, h_bev) for which the calibration takes place
    :return:
    """

    # tuple of marketing strengths (h_hev, h_phev, h_bev) for which the calibration takes place
    (h_hev_cal, h_phev_cal, h_bev_cal) = hs_cal

    # target adoption levels for hev, phev, and bev
    adoption_target = [0.489, 0.319, 0.192]

    # size of the search calibration parameter space
    num_parms = len(alpha_phevs) * len(alpha_bevs)

    # creating parameter space
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
        # for each calibration parameter set form the calibration parameter space, run simulations
        folder_name = get_folder_name(model_name=model_name,
                                      alpha_phev=alpha_phev,
                                      alpha_bev=alpha_bev,
                                      hs=(h_hev_cal, h_phev_cal, h_bev_cal),
                                      data_dir=data_dir)
        run_single_simulation(network_params=network_params,
                              heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                              heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                              heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                              heterogeneous_driving_patterns=heterogeneous_driving_patterns,
                              alpha_phev=alpha_phev,
                              alpha_bev=alpha_bev,
                              h_hev=h_hev_cal,
                              h_phev=h_phev_cal,
                              h_bev=h_bev_cal,
                              num_ave=500,
                              time_horizon=20,
                              folder_name=folder_name)
        sim_numbers = np.arange(int(np.loadtxt(folder_name + "/num_ave.csv")))

        # calculating the mean square error (MSE), see Section 4.5. Model calibration procedures
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

    # finding the best calibration parameters
    best_parms = min(error_dict, key=error_dict.get)
    print("Best parameters:", best_parms, "MSE:", error_dict[best_parms])

    alpha_phevs_m, alpha_bevs_m = np.meshgrid(alpha_phevs, alpha_bevs)
    errors = np.zeros(alpha_phevs_m.shape)

    for in1, i1 in enumerate(alpha_phevs):
        for in2, i2 in enumerate(alpha_bevs):
            errors[in2, in1] = error_dict[(i1, i2)]

    # saving the date from the calibration
    folder_name = data_dir + "/map_results"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    np.savetxt(folder_name + f"/{model_name}_mses.csv", errors, fmt="%.18f", delimiter=", ")
    np.savetxt(folder_name + f"/{model_name}_alpha_phevs_m.csv", alpha_phevs_m, fmt="%.18f", delimiter=",")
    np.savetxt(folder_name + f"/{model_name}_alpha_bevs_m.csv", alpha_bevs_m, fmt="%.18f", delimiter=",")

    # saving the best calibration parameters
    best_parms = (best_parms[0], best_parms[1], error_dict[best_parms])
    np.savetxt(folder_name + f"/{model_name}_best_parms.csv", best_parms, fmt="%.18f", delimiter=",")


def run_diagram_simulations(cal_data_dir, data_dir, network_params, heterogeneous_hev_susceptibilities,
                            heterogeneous_phev_susceptibilities, heterogeneous_bev_susceptibilities,
                            heterogeneous_driving_patterns):
    """
    runs simulations that are used to create diagrams illustrating how changing the strength of marketing
    of one type of AFV impacts the adoption levels of different AFVs
    the calibration parameters come from the grid-search calibration and are loaded from cal_data_dir
    :param cal_data_dir: path to the folder where the calibration data is saved
    :param data_dir: path to the folder where the diagram data is saved
    :param network_params: tuple of network parameters
    :param heterogeneous_hev_susceptibilities: 1 if hev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_phev_susceptibilities: 1 if phev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_bev_susceptibilities: 1 if bev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_driving_patterns: 1 if driving patterns are heterogeneous, 0 if homogeneous
    :return:
    """
    model_name = get_model_name(network_name=get_network_name(network_params),
                                heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                                heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                                heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                                heterogeneous_driving_patterns=heterogeneous_driving_patterns)

    # loading calibration parameters
    alpha_phev, alpha_bev, mse = np.loadtxt(cal_data_dir + f"/map_results/{model_name}_best_parms.csv")

    print(
        f"{network_params} {heterogeneous_hev_susceptibilities}_{heterogeneous_phev_susceptibilities}_{heterogeneous_bev_susceptibilities}hs_{heterogeneous_driving_patterns}hdp:\ta_phev={alpha_phev}\ta_bev={alpha_bev}\tMSE={mse}")

    # creating values of marketing strengths for which the simulations are carried out
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

        # running simulations with given parameters
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


def run_diagram_simulations_hyp(cal_data_dir, data_dir, network_params, heterogeneous_hev_susceptibilities,
                                heterogeneous_phev_susceptibilities, heterogeneous_bev_susceptibilities,
                                heterogeneous_driving_patterns):
    """
    runs simulations that are used to create diagrams illustrating how changing the strength of marketing
    of one type of AFV impacts the adoption levels of different AFVs
    the calibration parameters come from the tree-structured Parzen estimator algorithm and are loaded from cal_data_dir
    :param cal_data_dir: path to the folder where the calibration data is saved
    :param data_dir: path to the folder where the diagram data is saved
    :param network_params: tuple of network parameters
    :param heterogeneous_hev_susceptibilities: 1 if hev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_phev_susceptibilities: 1 if phev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_bev_susceptibilities: 1 if bev susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_driving_patterns: 1 if driving patterns are heterogeneous, 0 if homogeneous
    :return:
    """
    model_name = get_model_name(network_name=get_network_name(network_params),
                                heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                                heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                                heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                                heterogeneous_driving_patterns=heterogeneous_driving_patterns)
    print(model_name)

    # loading calibration parameters
    alpha_phev, alpha_bev, ch_hev, ch_phev, ch_bev, mse = np.loadtxt(cal_data_dir + f"/{model_name}_best_parms.csv")

    print(
        f"{network_params} {heterogeneous_hev_susceptibilities}_{heterogeneous_phev_susceptibilities}_{heterogeneous_bev_susceptibilities}hs_{heterogeneous_driving_patterns}hdp:\ta_phev={alpha_phev}\ta_bev={alpha_bev}\tMSE={mse}")

    # creating values of marketing strengths for which the simulations are carried out
    num_h = 20
    h_start = 0
    h_end = 4
    hs = np.linspace(h_start, h_end, num_h)
    h_inds = np.arange(num_h)

    for ih in range(3):
        if ih == 0:
            # hev
            h_inds_tuples = [(h, 0, 0) for h in h_inds]
            h_tuples = [(h, ch_phev, ch_bev) for h in hs]
        elif ih == 1:
            # phev
            h_inds_tuples = [(0, h, 0) for h in h_inds]
            h_tuples = [(ch_hev, h, ch_bev) for h in hs]
        else:
            # bev
            h_inds_tuples = [(0, 0, h) for h in h_inds]
            h_tuples = [(ch_hev, ch_phev, h) for h in hs]

        fdata_dir = data_dir + "/" + model_name + f"/{ih}"
        if not os.path.exists(fdata_dir):
            os.makedirs(fdata_dir)
        np.savetxt(fdata_dir + '/h_values.csv', np.column_stack((h_inds, hs)), fmt=("%i", "%.18f"))

        start = time.time()
        counter = 0
        for ind, hs_inds in enumerate(h_inds_tuples):
            (h_hev, h_phev, h_bev) = h_tuples[ind]
            folder_name = get_folder_name(model_name=model_name,
                                          alpha_phev=alpha_phev,
                                          alpha_bev=alpha_bev,
                                          hs=hs_inds,
                                          data_dir=fdata_dir)
            # running simulations
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
            print(f"{hs_inds}\t{h_tuples[ind]}\testimated remaining time:\t" + str(
                datetime.timedelta(seconds=int((len(h_inds_tuples) * 3 - counter) * ave_time))))


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
    sem = np.std(data, axis=0, ddof=1) / np.sqrt(np.size(data, axis=0))  # standard error of the mean
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
    """
    prepares data for plotting the diagrams
    """
    model_name = get_model_name(network_name=get_network_name(network_params),
                                heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                                heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                                heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                                heterogeneous_driving_patterns=heterogeneous_driving_patterns)
    alpha_phev, alpha_bev, ch_hev, ch_phev, ch_bev, mse = np.loadtxt(cal_data_dir + f"/{model_name}_best_parms.csv")
    if h_type == "h_hev":
        f_number = 0
    elif h_type == "h_phev":
        f_number = 1
    else:
        f_number = 2

    data_dir_path = data_dir + "/" + model_name + f"/{f_number}"
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
    """
    creates calibration maps: 3d plots illustrating how MSE changes for different values of the calibration parameters
    from the search-grid calibration
    """
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
    """
    creates plot with the time evolution of the adoption levels of AFVs
    """
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
    """
    creates plot with the time evolution of the adoption levels of AFVs
    """
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
    """
    creates diagram illustrating how changing a given marketing strength impacts the adoption levels of AFVs
    """
    result_folder_name = data_dir + f"/diagram_data"
    network_name = get_network_name(network_params=network_params)
    title_string = ""
    mark_it = 0
    marks = ["d", ".", "^", "x"]
    car_type = ["HEV", "PHEV", "BEV"]
    col_tab = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 6))
    counter = 0
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
            alpha_phev, alpha_bev, ch_hev, ch_phev, ch_bev, mse = np.loadtxt(
                cal_data_dir + f"/{model_name}_best_parms.csv")
            print(alpha_phev, alpha_bev, ch_hev, ch_phev, ch_bev, mse)

            title_string += f"{model_name} aphev={alpha_phev:.4f} abev={alpha_bev:.4f} {h_type}\n"
            data = np.loadtxt(result_folder_name + "/" + model_name + "_" + h_type + ".csv", delimiter=',')

            for i in range(3):
                if h_type == "h_hev":
                    ax[i].plot([ch_hev, ch_hev], [0, 1], color=f"{col_tab[counter]}")
                elif h_type == "h_phev":
                    ax[i].plot([ch_phev, ch_phev], [0, 1], color=f"{col_tab[counter]}")
                elif h_type == "h_bev":
                    ax[i].plot([ch_bev, ch_bev], [0, 1], color=f"{col_tab[counter]}")
                ax[i].errorbar(data[:, 0],
                               data[:, 1 + i * 2],
                               yerr=data[:, 2 + i * 2],
                               fmt="d",
                               markersize=5,
                               label=f"{heterogeneous_hev_susceptibilities}_{heterogeneous_phev_susceptibilities}_{heterogeneous_bev_susceptibilities}hs_{heterogeneous_driving_patterns}hdp")
                # ax[i].plot(data[:, 0],
                #            data[:, 1 + i * 2],
                #            markersize=5)
                ax[i].set_ylim([0, 1])
                ax[i].set_xlabel(h_type)
                ax[i].set_ylabel(f"Adoption level {car_type[i]}")
                # ax[i].set_title(car_type[i])
            counter += 1
    plt.suptitle(title_string)
    ax[0].plot([0, 4], [0.489] * 2, 'black')
    ax[1].plot([0, 4], [0.319] * 2, 'black')
    ax[2].plot([0, 4], [0.192] * 2, 'black')
    # plt.legend(loc="upper right")
    plt.tight_layout()

    plot_folder_name = data_dir + f"/diagram_plots"
    if not os.path.exists(plot_folder_name):
        os.mkdir(plot_folder_name)
    print(plot_folder_name + "/" + network_name + f"_{h_type}" + ".tex")
    tikzplotlib.save(plot_folder_name + "/" + network_name + f"_{h_type}" + ".tex", axis_height='\\figH',
                     axis_width='\\figW')
    # plt.savefig(plot_folder_name + "/" + network_name + f"_{h_type}" + ".png")
    # plt.show()


def objective_hyperopt(params, network_params, heterogeneous_hev_susceptibilities, heterogeneous_phev_susceptibilities,
                       heterogeneous_bev_susceptibilities, heterogeneous_driving_patterns, data_dir):
    """
    loss function used for the tree-structured Parzen estimator algorithm
    :return: the mean squared error (MSE)
    """
    alpha_phev, alpha_bev, h_hev, h_phev, h_bev = params['alpha_phev'], params['alpha_bev'], params[
        'h_hev'], params['h_phev'], params['h_bev']
    adoption_target = [0.489, 0.319, 0.192]

    model_name = get_model_name(network_name=get_network_name(network_params),
                                heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
                                heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
                                heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
                                heterogeneous_driving_patterns=heterogeneous_driving_patterns)

    folder_name = get_folder_name(model_name=model_name,
                                  alpha_phev=alpha_phev,
                                  alpha_bev=alpha_bev,
                                  hs=(h_hev, h_phev, h_bev),
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
    sim_numbers = np.arange(int(np.loadtxt(folder_name + "/num_ave.csv")))
    total_error = 0
    for sim_number in sim_numbers:
        file_name = folder_name + f"/sim-{sim_number}.txt"
        adoption_sim = np.loadtxt(file_name)[-1, :3]
        error = error_measure(adoption_target, adoption_sim)
        total_error += error
    total_error /= len(sim_numbers)
    return {'loss': total_error, 'status': STATUS_OK}


def run_automatedhyp_calibration(hypcal_data_dir, network_params, heterogeneous_susceptibilities,
                                 heterogeneous_driving_patterns):
    """
    runs a model calibration based on the tree-structured Parzen estimator algorithm
    :param hypcal_data_dir: path to the folder where the calibration data is saved
    :param network_params: tuple of network parameters
    :param heterogeneous_susceptibilities: 1 if susceptibilities are heterogeneous, 0 if homogeneous
    :param heterogeneous_driving_patterns: 1 if driving patterns are heterogeneous, 0 if homogeneous
    :return:
    """
    objective = partial(objective_hyperopt,
                        network_params=network_params,
                        heterogeneous_hev_susceptibilities=heterogeneous_susceptibilities,
                        heterogeneous_phev_susceptibilities=heterogeneous_susceptibilities,
                        heterogeneous_bev_susceptibilities=heterogeneous_susceptibilities,
                        heterogeneous_driving_patterns=heterogeneous_driving_patterns,
                        data_dir=hypcal_data_dir)

    # define the calibration parameter space
    space = {
        'alpha_phev': hp.uniform('alpha_phev', 0, 14),
        'alpha_bev': hp.uniform('alpha_bev', 0, 1.5),
        'h_hev': hp.uniform('h_hev', 0, 4),
        'h_phev': hp.uniform('h_phev', 0, 4),
        'h_bev': hp.uniform('h_bev', 0, 4)
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=1000,
        trials=trials
    )
    print(best)
    print('MSE:', trials.best_trial['result']['loss'])
    model_name = get_model_name(network_name=get_network_name(network_params),
                                heterogeneous_hev_susceptibilities=heterogeneous_susceptibilities,
                                heterogeneous_phev_susceptibilities=heterogeneous_susceptibilities,
                                heterogeneous_bev_susceptibilities=heterogeneous_susceptibilities,
                                heterogeneous_driving_patterns=heterogeneous_driving_patterns)
    best_parms = (best['alpha_phev'], best['alpha_bev'], best['h_hev'], best['h_phev'], best['h_bev'],
                  trials.best_trial['result']['loss'])
    np.savetxt(hypcal_data_dir + f"/{model_name}_best_parms.csv", best_parms, fmt="%.18f", delimiter=",")
