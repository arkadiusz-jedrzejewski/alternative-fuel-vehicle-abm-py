import os

import matplotlib.pyplot as plt
import numpy as np
from afv_module import run_automated_calibration, run_diagram_simulations, get_mean_and_quantiles, get_diagram_data, \
    get_calibration, get_diagram

if __name__ == '__main__':

    cal_data_dir = r"D:\Data\alternative-fuel-vehicle-abm-cal-data-3"
    if not os.path.exists(cal_data_dir):
        os.mkdir(cal_data_dir)

    data_dir = r"D:\Data\alternative-fuel-vehicle-abm-data-3"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    alpha_phevs = np.linspace(4, 15, 12)
    alpha_bevs = np.linspace(0, 1.5, 16)
    networks = [("SL", 32),
                ("WS", 1024, 4, 1),
                ("WS", 1024, 4, 0)]

    #get_calibration(cal_data_dir, networks[0], 0, 1)
    get_diagram(cal_data_dir=cal_data_dir,
                data_dir=data_dir,
                network_params=networks[0],
                h_type="h_phev")

    # print(result[0][0])

    #plt.figure()
    #fig, ax = plt.subplots(1, 3)
    # for network_params in networks:
    #     for heterogeneous_susceptibilities in [0, 1]:
    #         for heterogeneous_driving_patterns in [0, 1]:
    #             for h_type in ["h_hev", "h_phev", "h_bev"]:
    #                 data = get_diagram_data(cal_data_dir=cal_data_dir,
    #                                         data_dir=data_dir,
    #                                         network_params=network_params,
    #                                         heterogeneous_susceptibilities=heterogeneous_susceptibilities,
    #                                         heterogeneous_driving_patterns=heterogeneous_driving_patterns,
    #                                         h_type=h_type)


            # ax[0].errorbar(h_values, hevs, yerr=sem_hevs, fmt=':.')
            # ax[0].set_ylim([0, 1])
            # #plt.plot(h_values, hevs, '--.')
            # ax[1].errorbar(h_values, phevs, yerr=sem_phevs, fmt=':.')
            # ax[1].set_ylim([0, 1])
            # #plt.plot(h_values, phevs, '--.')
            # ax[2].errorbar(h_values, bevs, yerr=sem_bevs, fmt=':.', label=f"{heterogeneous_susceptibilities}hs_{heterogeneous_driving_patterns}hdp")
            # ax[2].set_ylim([0, 1])
            #plt.plot(h_values, bevs, '--.')
            #plt.errorbar(h_values, nones, yerr=sem_nones, fmt=':.')
            #plt.plot(h_values, nones, '--.')
    # plt.legend(loc="upper right")
    # plt.show()
    # for network_params in networks:
    #     for heterogeneous_susceptibilities in [0, 1]:
    #         for heterogeneous_driving_patterns in [0, 1]:
    #             get_calibration(cal_data_dir=cal_data_dir,
    #                             network_params=network_params,
    #                             heterogeneous_susceptibilities=heterogeneous_susceptibilities,
    #                             heterogeneous_driving_patterns=heterogeneous_driving_patterns)

    #             run_automated_calibration(data_dir=cal_data_dir,
    #                                       alpha_phevs=alpha_phevs,
    #                                       alpha_bevs=alpha_bevs,
    #                                       network_params=network_params,
    #                                       heterogeneous_susceptibilities=heterogeneous_susceptibilities,
    #                                       heterogeneous_driving_patterns=heterogeneous_driving_patterns)
    #             run_diagram_simulations(cal_data_dir=cal_data_dir,
    #                                     data_dir=data_dir,
    #                                     network_params=network_params,
    #                                     heterogeneous_susceptibilities=heterogeneous_susceptibilities,
    #                                     heterogeneous_driving_patterns=heterogeneous_driving_patterns)
