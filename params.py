from math import pi
import numpy as np


# FOG parameters
FOG_ROTATION = np.array([[1., 0., 0.], 
                         [0., 1., 0.],
                         [0., 0., 1.]])
FOG_POSITION = np.array([-0.335, -0.035, 0.78])

# LiDAR parameters
LIDAR_ANGLES = np.linspace(-5, 185, num=286) / 360 * 2 * pi
LIDAR_ANGLE_COS = np.cos(LIDAR_ANGLES)
LIDAR_ANGLE_SIN = np.sin(LIDAR_ANGLES)
LIDAR_MAXRANGE = 80
LIDAR_ROTATION = np.array([[0.00130201, 0.796097, 0.605167], 
                           [0.999999, -0.000419027, -0.00160026], 
                           [-0.00102038, 0.605169, -0.796097]])
LIDAR_POSITION = np.array([0.8349, -0.0126869, 1.76416])
N_LIDAR_SAMPLES = 4

# Encoder parameters
ENCODER_RESOLUTION = 4096
ENCODER_LEFT_DIAMETER = 0.623479
ENCODER_RIGHT_DIAMETER = 0.622806
ENCODER_WHEEL_BASE = 1.52439

# Stereo camara parameters
STEREO_BASELINE = 0.475143600050775
STEREO_ROTATION = np.array([[-0.00680499, -0.0153215, 0.99985   ],
                            [-0.999977, 0.000334627, -0.00680066],
                            [-0.000230383, -0.999883, -0.0153234]])
STEREO_POSITION = np.array([1.64239, 0.247401, 1.58411])

# General parameters
INIT_LOGODDS = 0
LOGODDS_LOWER = -10
LOGODDS_UPPER = 10
LOGODDS_UPDATE = np.log(4)
NUM_PARTICLES = 30
STEPS_FIGURES = 5000
VELOCITY_NOISE = 1e-10
A_VELOCITY_NOISE = 1e-12
MAP_SIZE = [-100, 1400, -1000, 100]
MAP_RESOLUTION = 1/3
RESAMPLE_THRESHOLD = 0.1