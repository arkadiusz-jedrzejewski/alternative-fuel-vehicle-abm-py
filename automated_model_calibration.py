import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from afv_module import run_automated_calibration, run_diagram_simulations, get_mean_and_quantiles, get_diagram_data, \
    get_calibration, get_diagram, get_calibration_map, get_calibration_ax, objective_hyperopt, get_network_name, \
    get_model_name, run_automatedhyp_calibration, run_diagram_simulations_hyp
import tikzplotlib
from hyperopt import tpe, hp, fmin, Trials

if __name__ == '__main__':

    # folder where the calibration data is saved based on the grid search
    cal_data_dir = r"D:\Data\alternative-fuel-vehicle-abm-cal-data-1"
    if not os.path.exists(cal_data_dir):
        os.mkdir(cal_data_dir)

    # folder where data for the adoption rate diagrams is saved based on the grid search
    data_dir = r"D:\Data\alternative-fuel-vehicle-abm-data-1"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # parameter space for srid search
    alpha_phevs = np.linspace(0, 15, 16)
    alpha_bevs = np.linspace(0, 1.5, 16)

    # parameters of the networks
    # ("SL", L)             - Square Lattice with N = L x L agents
    # ("WS", N, k, beta)    - Watts-Strogatz network:
    #                           N - number of agents
    #                           k - average node degree
    #                           beta - rewiring probability
    networks = [("SL", 32),
                ("WS", 1024, 4, 1),
                ("WS", 1024, 4, 0)]

    # for network_params in networks:
    #     for heterogeneous_driving_patterns in [0, 1]:
    #         for heterogeneous_susceptibilities in [0, 1]:
    #             # run_automated_calibration(data_dir=cal_data_dir,
    #             #                           alpha_phevs=alpha_phevs,
    #             #                           alpha_bevs=alpha_bevs,
    #             #                           network_params=network_params,
    #             #                           heterogeneous_hev_susceptibilities=heterogeneous_susceptibilities,
    #             #                           heterogeneous_phev_susceptibilities=heterogeneous_susceptibilities,
    #             #                           heterogeneous_bev_susceptibilities=heterogeneous_susceptibilities,
    #             #                           heterogeneous_driving_patterns=heterogeneous_driving_patterns,
    #             #                           hs_cal=(0, 0, 2))
    #             get_calibration_map(cal_data_dir, network_params, heterogeneous_susceptibilities,
    #                                 heterogeneous_susceptibilities, heterogeneous_susceptibilities,
    #                                 heterogeneous_driving_patterns)

    # for network_params in networks:
    #     for heterogeneous_susceptibilities in [0, 1]:
    #         for heterogeneous_driving_patterns in [0, 1]:
    #             for h_type in ["h_hev", "h_phev", "h_bev"]:
    #                 run_diagram_simulations(cal_data_dir=cal_data_dir,
    #                                         data_dir=data_dir,
    #                                         network_params=network_params,
    #                                         heterogeneous_hev_susceptibilities=heterogeneous_susceptibilities,
    #                                         heterogeneous_phev_susceptibilities=heterogeneous_susceptibilities,
    #                                         heterogeneous_bev_susceptibilities=heterogeneous_susceptibilities,
    #                                         heterogeneous_driving_patterns=heterogeneous_driving_patterns)

    # folder where the calibration data is saved based on the tree-structured Parzen estimator algorithm
    hypcal_data_dir = r"D:\Data\alternative-fuel-vehicle-abm-hypcal-data-2"
    if not os.path.exists(hypcal_data_dir):
        os.mkdir(hypcal_data_dir)

    # for network_params in networks:
    #     for heterogeneous_driving_patterns in [0, 1]:
    #         for heterogeneous_susceptibilities in [0, 1]:
    #             run_automatedhyp_calibration(hypcal_data_dir=hypcal_data_dir,
    #                                          network_params=network_params,
    #                                          heterogeneous_driving_patterns=heterogeneous_driving_patterns,
    #                                          heterogeneous_susceptibilities=heterogeneous_susceptibilities)

    # folder where data for the adoption rate diagrams is saved based on the tree-structured Parzen estimator algorithm
    hypdata_dir = r"D:\Data\alternative-fuel-vehicle-abm-hyp-data-4"
    if not os.path.exists(hypdata_dir):
        os.mkdir(hypdata_dir)

    # for network_params in networks:
    #     for heterogeneous_driving_patterns in [0, 1]:
    #         for heterogeneous_susceptibilities in [0, 1]:
    #             run_diagram_simulations_hyp(cal_data_dir=hypcal_data_dir,
    #                                         data_dir=hypdata_dir,
    #                                         network_params=network_params,
    #                                         heterogeneous_hev_susceptibilities=heterogeneous_susceptibilities,
    #                                         heterogeneous_phev_susceptibilities=heterogeneous_susceptibilities,
    #                                         heterogeneous_bev_susceptibilities=heterogeneous_susceptibilities,
    #                                         heterogeneous_driving_patterns=heterogeneous_driving_patterns)

    # get_calibration_map(cal_data_dir, networks[0], 0, 1)
    # plt.show()
    # get_calibration(cal_data_dir, networks[0], 0, 1)

    # for network_params in networks:
    #     for heterogeneous_susceptibilities in [0, 1]:
    #         for heterogeneous_driving_patterns in [0, 1]:
    #             for h_type in ["h_hev", "h_phev", "h_bev"]:
    #                 data = get_diagram_data(cal_data_dir=hypcal_data_dir,
    #                                         data_dir=hypdata_dir,
    #                                         network_params=network_params,
    #                                         heterogeneous_hev_susceptibilities=heterogeneous_susceptibilities,
    #                                         heterogeneous_phev_susceptibilities=heterogeneous_susceptibilities,
    #                                         heterogeneous_bev_susceptibilities=heterogeneous_susceptibilities,
    #                                         heterogeneous_driving_patterns=heterogeneous_driving_patterns,
    #                                         h_type=h_type)
    for network_params in networks:
        for h_type in ["h_hev", "h_phev", "h_bev"]:
            get_diagram(cal_data_dir=hypcal_data_dir,
                        data_dir=hypdata_dir,
                        network_params=network_params,
                        h_type=h_type)
    #     plt.show()

    # print(result[0][0])

    # plt.figure()
    # fig, ax = plt.subplots(1, 3)
    # for network_params in networks:
    #         for heterogeneous_hev_susceptibilities in [0, 1]:
    #             for heterogeneous_phev_susceptibilities in [0, 1]:
    #                 for heterogeneous_bev_susceptibilities in [0, 1]:
    #                     for heterogeneous_driving_patterns in [0, 1]:
    #                         for h_type in ["h_hev", "h_phev", "h_bev"]:
    #                             data = get_diagram_data(cal_data_dir=cal_data_dir,
    #                                                     data_dir=data_dir,
    #                                                     network_params=network_params,
    #                                                     heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
    #                                                     heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
    #                                                     heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
    #                                                     heterogeneous_driving_patterns=heterogeneous_driving_patterns,
    #                                                     h_type=h_type)

    # ax[0].errorbar(h_values, hevs, yerr=sem_hevs, fmt=':.')
    # ax[0].set_ylim([0, 1])
    # #plt.plot(h_values, hevs, '--.')
    # ax[1].errorbar(h_values, phevs, yerr=sem_phevs, fmt=':.')
    # ax[1].set_ylim([0, 1])
    # #plt.plot(h_values, phevs, '--.')
    # ax[2].errorbar(h_values, bevs, yerr=sem_bevs, fmt=':.', label=f"{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp")
    # ax[2].set_ylim([0, 1])
    # plt.plot(h_values, bevs, '--.')
    # plt.errorbar(h_values, nones, yerr=sem_nones, fmt=':.')
    # plt.plot(h_values, nones, '--.')
    # plt.legend(loc="upper right")
    # plt.show()

    # fig, ax = plt.subplots(2, 2)
    # get_calibration_ax(
    #     ax=ax[0, 0],
    #     cal_data_dir=cal_data_dir,
    #     network_params=("WS", 1024, 4, 0),
    #     heterogeneous_hev_susceptibilities=0,
    #     heterogeneous_phev_susceptibilities=0,
    #     heterogeneous_bev_susceptibilities=0,
    #     heterogeneous_driving_patterns=0)
    # get_calibration_ax(
    #     ax=ax[0, 1],
    #     cal_data_dir=cal_data_dir,
    #     network_params=("WS", 1024, 4, 0),
    #     heterogeneous_hev_susceptibilities=0,
    #     heterogeneous_phev_susceptibilities=0,
    #     heterogeneous_bev_susceptibilities=0,
    #     heterogeneous_driving_patterns=1)
    # get_calibration_ax(
    #     ax=ax[1, 0],
    #     cal_data_dir=cal_data_dir,
    #     network_params=("WS", 1024, 4, 0),
    #     heterogeneous_hev_susceptibilities=1,
    #     heterogeneous_phev_susceptibilities=1,
    #     heterogeneous_bev_susceptibilities=1,
    #     heterogeneous_driving_patterns=0)
    # save_path = get_calibration_ax(
    #     ax=ax[1, 1],
    #     cal_data_dir=cal_data_dir,
    #     network_params=("WS", 1024, 4, 0),
    #     heterogeneous_hev_susceptibilities=1,
    #     heterogeneous_phev_susceptibilities=1,
    #     heterogeneous_bev_susceptibilities=1,
    #     heterogeneous_driving_patterns=1)
    # plt.tight_layout()
    #
    # tikzplotlib.save(save_path, axis_height='\\figH', axis_width='\\figW')
    # plt.show()

    # for network_params in networks:
    #     for heterogeneous_hev_susceptibilities in [0, 1]:
    #         for heterogeneous_phev_susceptibilities in [0, 1]:
    #             for heterogeneous_bev_susceptibilities in [0, 1]:
    #                 for heterogeneous_driving_patterns in [0, 1]:
    #                     get_calibration_map(cal_data_dir,
    #                                         network_params=network_params,
    #                                         heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
    #                                         heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
    #                                         heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
    #                                         heterogeneous_driving_patterns=heterogeneous_driving_patterns)
    #                     plt.show()
    # get_calibration(cal_data_dir=cal_data_dir,
    #                 network_params=network_params,
    #                 heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
    #                 heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
    #                 heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
    #                 heterogeneous_driving_patterns=heterogeneous_driving_patterns)

    # run_diagram_simulations(cal_data_dir=cal_data_dir,
    #                         data_dir=data_dir,
    #                         network_params=network_params,
    #                         heterogeneous_hev_susceptibilities=heterogeneous_hev_susceptibilities,
    #                         heterogeneous_phev_susceptibilities=heterogeneous_phev_susceptibilities,
    #                         heterogeneous_bev_susceptibilities=heterogeneous_bev_susceptibilities,
    #                         heterogeneous_driving_patterns=heterogeneous_driving_patterns)
