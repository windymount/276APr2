import numpy as np
from mapping import physics2map
from params import LIDAR_ANGLE_COS, LIDAR_ANGLE_SIN, LIDAR_MAXRANGE, LIDAR_POSITION, LIDAR_ROTATION, N_LIDAR_SAMPLES, RESAMPLE_THRESHOLD, VELOCITY_NOISE, A_VELOCITY_NOISE
from pr2_utils import mapCorrelation, my_map_correlation, transform_2d_to_3d, transform_orient_to_mat


def create_particles(n):
    """
    
    Create paticles at (0, 0)
    
    :param int n: number of particles.
    :return position: particle position.
    :return orient: particle orientation.
    :return weight: weights of particles
    """
    position = np.zeros((2, n))
    orient = np.zeros((n, ))
    weight = np.ones((n, )) / n
    return position, orient, weight


def predict_particles(position, orient, velocity_data, a_velocity_data, time_step,
                      noise_v=VELOCITY_NOISE, noise_a=A_VELOCITY_NOISE):
    """
    
    Predict particle state using Ackermann-drive model.

    :param position: particle position.
    :param orient: particle orientation.
    :param velocity_data: velocity data from encoder
    :param a_velocity_data: angular velocity data from FOG
    :param float time_step: time difference of this prediction step
    :param float noise_v: Gaussian std variance of noise in velocity
    :param float noise_a: Gaussian std variance of noise in angular velocity
    :return predicted_position:
    :return predicted_orient:
    
    """
    # Add noise
    av = a_velocity_data[-1]
    velocity_data = velocity_data + np.random.normal(scale=noise_v, size=orient.shape)


    v_x = velocity_data * np.cos(orient)
    v_y = velocity_data * np.sin(orient)
    # Get the yaw angular velocity
    av = av + np.random.normal(scale=noise_a, size=orient.shape)
    # Update position
    position += np.vstack([v_x, v_y]) * time_step
    # Update orientation
    orient = orient + av * time_step

    return position, orient


def update_particles(position, orient, weights, lidar_data, map, xm, ym):
    """
    
    Update particle state according to lidar.

    :param position: particle position.
    :param orient: particle orientation.
    :param velocity_data: velocity data from encoder
    :param a_velocity_data: angular velocity data from FOG
    :param float time_step: time difference of this prediction step
    :return updated_weights:
    
    """
    # Calculate LiDAR sketch
    body_rotation = transform_orient_to_mat(orient)

    li2wo_rotation = body_rotation @ LIDAR_ROTATION[0:2, 0:2]
    li2wo_position = position + LIDAR_POSITION[0:2, None]

    # Compute LiDAR endpoints
    co_li_x = lidar_data * LIDAR_ANGLE_COS
    co_li_y = lidar_data * LIDAR_ANGLE_SIN
    max_ranges = np.where(lidar_data == LIDAR_MAXRANGE)[0]
    # Calculate world coordinate: 2 * N_particles * N_LIDAR_SCAN
    co_wo = li2wo_position[:, :, None] + np.reshape(li2wo_rotation @ np.vstack([co_li_x, co_li_y]), newshape=(-1, 2, len(LIDAR_ANGLE_COS))).transpose((1, 0, 2))
    particle_weights = np.zeros(co_wo.shape[1])
    # Iterate over particles
    for particle in range(co_wo.shape[1]):
        p_co_wo = co_wo[:, particle, :]
        x_co = np.linspace(np.zeros_like(p_co_wo[0, :]), p_co_wo[0, :], num=N_LIDAR_SAMPLES)
        y_co = np.linspace(np.zeros_like(p_co_wo[1, :]), p_co_wo[1, :], num=N_LIDAR_SAMPLES)
        hit_obstacle = np.zeros_like(x_co)
        hit_obstacle[-1, :][max_ranges] = 1
        hit_obstacle = hit_obstacle.reshape(-1)
        particle_weights[particle] = my_map_correlation(map, xm, ym, np.vstack([x_co.reshape(-1), y_co.reshape(-1)]), hit_obstacle)
    # particle_weights /= len(hit_obstacle)
    updated_weights = (weights * np.exp(particle_weights-np.max(particle_weights)))
    updated_weights /= np.sum(updated_weights)
    return updated_weights


def resample_particles(position, orient, weights):
    # Resample particles if effective particles too few.
    n_particles = len(weights)
    eff_particles = 1 / np.sum(weights ** 2)
    if eff_particles / n_particles < RESAMPLE_THRESHOLD:
        idxes = np.random.choice(range(n_particles), size=n_particles, replace=True, p=weights)
        return position[:, idxes], orient[idxes], np.ones_like(weights) / n_particles
    else:
        return position, orient, weights



