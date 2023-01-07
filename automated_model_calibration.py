import os
import numpy as np
from afv_module import run_automated_calibration

if __name__ == '__main__':

    data_dir = r"D:\Data\alternative-fuel-vehicle-abm-data-2"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    alpha_phevs = np.linspace(4, 15, 2)
    alpha_bevs = np.linspace(0, 1.5, 2)
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
