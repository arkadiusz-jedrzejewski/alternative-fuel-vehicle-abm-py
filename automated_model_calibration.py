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

    # parameters of the networks
    # ("SL", L)             - Square Lattice with N = L x L agents
    # ("WS", N, k, beta)    - Watts-Strogatz network:
    #                           N - number of agents
    #                           k - average node degree
    #                           beta - rewiring probability
    networks = [("SL", 32),
                ("WS", 1024, 4, 1),
                ("WS", 1024, 4, 0)]

    #  run calibration based on the grid search
    # folder where the calibration data is saved based on the grid search
    cal_data_dir = r"D:\Data\alternative-fuel-vehicle-abm-cal-data-1"
    if not os.path.exists(cal_data_dir):
        os.mkdir(cal_data_dir)

    # folder where data for the adoption rate diagrams is saved based on the grid search calibration
    data_dir = r"D:\Data\alternative-fuel-vehicle-abm-data-1"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # parameter space for srid search
    alpha_phevs = np.linspace(0, 15, 16)
    alpha_bevs = np.linspace(0, 1.5, 16)

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

    # run simulations for the adoption rate diagrams based on the grid search calibration
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
    #
    # end of grid-search block

    # run calibration based on the tree-structured Parzen estimator algorithm
    # folder where the calibration data is saved based on the tree-structured Parzen estimator algorithm
    hypcal_data_dir = r"D:\Data\alternative-fuel-vehicle-abm-hypcal-data-2"
    if not os.path.exists(hypcal_data_dir):
        os.mkdir(hypcal_data_dir)

    # run calibration based on the tree-structured Parzen estimator algorithm
    # for network_params in networks:
    #     for heterogeneous_driving_patterns in [0, 1]:
    #         for heterogeneous_susceptibilities in [0, 1]:
    #             run_automatedhyp_calibration(hypcal_data_dir=hypcal_data_dir,
    #                                          network_params=network_params,
    #                                          heterogeneous_driving_patterns=heterogeneous_driving_patterns,
    #                                          heterogeneous_susceptibilities=heterogeneous_susceptibilities)

    # folder where data for the adoption rate diagrams is saved
    # based on the tree-structured Parzen estimator algorithm calibration
    hypdata_dir = r"D:\Data\alternative-fuel-vehicle-abm-hyp-data-4"
    if not os.path.exists(hypdata_dir):
        os.mkdir(hypdata_dir)

    # run simulations for the adoption rate diagrams based on the tree-structured Parzen estimator algorithm calibration
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

    for network_params in networks:
        for h_type in ["h_hev", "h_phev", "h_bev"]:
            get_diagram(cal_data_dir=hypcal_data_dir,
                        data_dir=hypdata_dir,
                        network_params=network_params,
                        h_type=h_type)

