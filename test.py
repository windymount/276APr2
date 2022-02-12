import mapping
from matplotlib import pyplot as plt
from pr2_utils import read_data_from_csv, lidar_correction
import numpy as np


def test_init_mapping():
    map, xm, ym = mapping.create_map(-100, 100, -100, 100, 1)
    plt.imshow(mapping.map2prob(map))
    lidar_time, lidar_data = read_data_from_csv("data/sensor_data/lidar.csv")
    init_rotate = np.eye(3)
    init_position = np.zeros(3)
    map = mapping.update_map(map, xm, ym, init_rotate, init_position, lidar_correction(lidar_data[0, :]))
    plt.imshow(mapping.map2prob(map))
    plt.show(block=True)


if __name__ == "__main__":
    test_init_mapping()