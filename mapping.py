import numpy as np
from params import INIT_LOGODDS, LIDAR_ANGLE_COS, LIDAR_ANGLE_SIN, LIDAR_MAXRANGE, LIDAR_POSITION, LIDAR_ROTATION, LOGODDS_LOWER, LOGODDS_UPDATE, LOGODDS_UPPER
from pr2_utils import bresenham2D, physics2map


def create_map(x_begin, x_end, y_begin, y_end, grid_size):
    """
    
    Initialize map.

    :param float x_begin: begin of x-axis (physical)
    :param float x_end: end of x-axis (physical)
    :param float y_begin: begin of x-axis (physical)
    :param float y_end: end of y-axis (physical)
    :param float grid_size: length of grid edge
    :return map: occupancy map
    :return xm: x position (physical)
    :return ym: y position (physical)
    
    """
    xm = np.linspace(x_begin, x_end, int((x_end - x_begin)/grid_size) + 1)
    ym = np.linspace(y_begin, y_end, int((y_end - y_begin)/grid_size) + 1)
    map = np.zeros((xm.size, ym.size)) + INIT_LOGODDS
    return map, xm, ym


def map2prob(map):
    # Get occupied probability from map.
    prob = np.exp(map)
    return prob / (1 + prob)


def update_map(map, xm, ym, body_rotation, body_position, lidar_data):
    """
    
    Update occupancy map according to LiDAR data.

    :param map: occupancy map
    :param xm: x position of map (physical)
    :param ym: y position of map (physical)
    :param body_rotation: rotation matrix from body frame to world frame
    :param body_position: position vector from body frame to world frame
    :param lidar_data: lidar detection data
    :return updated_map:
    
    """
    # Compute LiDAR to world transform
    li2wo_rotation = body_rotation @ LIDAR_ROTATION
    li2wo_position = body_position + LIDAR_POSITION

    # Compute LiDAR endpoints
    co_li_x = lidar_data * LIDAR_ANGLE_COS
    co_li_y = lidar_data * LIDAR_ANGLE_SIN
    max_ranges = np.where(lidar_data == LIDAR_MAXRANGE)[0]
    co_wo = li2wo_position[:, None] + li2wo_rotation @ np.vstack([co_li_x, co_li_y, np.zeros_like(co_li_x)])
    Xidx, Yidx = physics2map(map, xm, ym, co_wo[0, :], co_wo[1, :])

    # Compute LiDAR startpoints
    X_start, Y_start = physics2map(map, xm, ym, li2wo_position[0], li2wo_position[1])
    for i, (x, y) in enumerate(zip(Xidx, Yidx)):
        path_idx = bresenham2D(X_start, Y_start, x, y)
        path_idx = path_idx.astype(np.int16)
        map[path_idx[0, :-1], path_idx[1, :-1]] -= LOGODDS_UPDATE
        if i not in max_ranges:
            map[path_idx[0, -1], path_idx[1, -1]] += LOGODDS_UPDATE
        map[path_idx[0, :], path_idx[1, :]] = np.minimum(np.maximum(map[path_idx[0, :], path_idx[1, :]], LOGODDS_LOWER), LOGODDS_UPPER)
    return map


def draw_map_from_stereo(map, xm, ym, stereo_img_l, disparity):
    """
    
    Mark point at map with RGB color.
    
    """