import numpy as np
from mapping import physics2map
from params import VELOCITY_NOISE, A_VELOCITY_NOISE


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
    weight = np.ones((n, ))
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
    velocity_data = velocity_data + np.random.normal(scale=noise_v, size=orient.shape)


    v_x = velocity_data * np.cos(orient)
    v_y = velocity_data * np.sin(orient)
    # Get the yaw angular velocity
    av = a_velocity_data[-1]
    av = av + np.random.normal(scale=noise_a, size=orient.shape)

    # Update position
    position += np.vstack([v_x, v_y]) * time_step
    # Update orientation
    orient = orient + av * time_step

    return position, orient







