import mapping
import particle_filter
from matplotlib import pyplot as plt
from pr2_utils import physics2map, read_data_from_csv, correct_lidar, get_angular_velocity, get_velocity, show_particles_on_map, transform_2d_to_3d
from params import MAP_RESOLUTION, MAP_SIZE, NUM_PARTICLES, STEPS_FIGURES
import numpy as np
import warnings
warnings.filterwarnings("error")


def main(n_particles):
    # Initialize map
    map, xm, ym = mapping.create_map(*MAP_SIZE, MAP_RESOLUTION)
    particle_map = np.zeros_like(map, dtype=np.int8)
    # Read and process sensor data
    lidar_time, lidar_data = read_data_from_csv("data/sensor_data/lidar.csv")
    lidar_data = correct_lidar(lidar_data)
    v_time, v_data = get_velocity("data/sensor_data/encoder.csv")
    a_time, a_data = get_angular_velocity("data/sensor_data/fog.csv")
    # Read disparity map
    disparity = np.load("data/disparity.npy")
    stereo_time = np.load("time_stamp.npy")
    # Create event series
    event_map = {}

    def add_to_map(time, event_map, event):
        for idx, t in enumerate(time):
            if t in event_map:
                event_map[t].append((event, idx))
            else:
                event_map[t] = [(event, idx)]
        return event_map

    event_map = add_to_map(lidar_time, event_map, "lidar")
    event_map = add_to_map(v_time, event_map, "v")
    event_map = add_to_map(a_time, event_map, "a")
    evnet_map = add_to_map(stereo_time, event_map, "stereo")
    timeline = sorted(event_map.keys())

    # Create particles
    p_position, p_orient, p_weight = particle_filter.create_particles(NUM_PARTICLES)
    p_v, p_av, last_predict = None, None, None
    for t_idx, t in enumerate(timeline):
        events = event_map[t]
        for event in events:
            event_type, event_idx = event
            if event_type == "v":
                # Update velocity
                p_v = v_data[event_idx]

                if (last_predict is None):
                    if (p_v is not None) and (p_av is not None):last_predict = t
                else:
                    # Do prediction
                    p_position, p_orient = particle_filter.predict_particles(p_position, p_orient, p_v, p_av, t-last_predict)
                    last_predict = t

            elif event_type == "a":
                # Update angular velocity
                p_av = a_data[event_idx]

                if (last_predict is None):
                    if (p_v is not None) and (p_av is not None):last_predict = t
                else:
                    # Do prediction
                    p_position, p_orient = particle_filter.predict_particles(p_position, p_orient, p_v, p_av, t-last_predict)
                    last_predict = t

            elif event_type == "lidar":
                # Record partcles
                xidx, yidx = physics2map(map, xm, ym, p_position[0, :], p_position[1, :])
                particle_map[xidx, yidx] = 1
                # Update map using lidar info and update particle weights
                observer_id = np.argmax(p_weight)
                position, rotation = transform_2d_to_3d(p_position[:, observer_id], p_orient[observer_id])
                c_lidar = lidar_data[event_idx, :]
                p_weight = particle_filter.update_particles(p_position, p_orient, p_weight, c_lidar, map, xm, ym)
                map = mapping.update_map(map, xm, ym, rotation, position, c_lidar)
                p_position, p_orient, p_weight = particle_filter.resample_particles(p_position, p_orient, p_weight)
            
            elif event_type == "stereo":
                pass
            # map = show_particles_on_map(map, xm, ym, p_position)
        if t_idx and t_idx % STEPS_FIGURES == 0: 
            plt.imshow(np.sign(map).T)
            xidx, yidx = physics2map(map, xm, ym, p_position[0, :], p_position[1, :])
            plt.scatter(np.where(), s=0.01, c='r')
            plt.savefig("img/step{}.png".format(t_idx), dpi=600)
            plt.cla() 
            plt.clf() 
            plt.close('all')
            # plt.show(block=True)

if __name__ == "__main__":
    main(NUM_PARTICLES)