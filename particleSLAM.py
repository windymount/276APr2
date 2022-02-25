import os
import cv2
import mapping
import particle_filter
from matplotlib import pyplot as plt
from pr2_utils import calculate_camera, physics2map, read_data_from_csv, correct_lidar, get_angular_velocity, get_velocity, recover_space_coordinate, show_particles_on_map, transform_2d_to_3d
from params import IMG_OUTPUT_PATH, MAP_RESOLUTION, MAP_SIZE, NUM_PARTICLES, STEPS_FIGURES, STEPS_TRAJECTORY, STEREO_POSITION, STEREO_ROTATION, STEREO_Z_RANGE, UPDATE_INTERVAL, MAPPING_INTERVAL
import numpy as np
import gc
import shutil


def main(n_particles):
    # Initialize map
    map, xm, ym = mapping.create_map(*MAP_SIZE, MAP_RESOLUTION)
    color_map = np.zeros((3, *map.shape), dtype=np.uint8)
    # Read and process sensor data
    lidar_time, lidar_data = read_data_from_csv("data/sensor_data/lidar.csv")
    lidar_data = correct_lidar(lidar_data)
    v_time, v_data = get_velocity("data/sensor_data/encoder.csv")
    a_time, a_data = get_angular_velocity("data/sensor_data/fog.csv")
    # Read disparity map
    disparity = np.load("data/disparity.npy")
    stereo_time = np.load("data/time_stamp.npy")
    camera_transition = calculate_camera()
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
    event_map = add_to_map(stereo_time, event_map, "stereo")
    timeline = sorted(event_map.keys())

    # Create particles
    p_position, p_orient, p_weight = particle_filter.create_particles(NUM_PARTICLES)
    p_v, p_av, last_predict = None, None, None
    traj_x, traj_y = [], []
    if not os.path.exists(IMG_OUTPUT_PATH):
        os.makedirs(IMG_OUTPUT_PATH)
    # Copy parameters
    shutil.copy("params.py", IMG_OUTPUT_PATH)
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
                # Update map using lidar info and update particle weights
                observer_id = np.argmax(p_weight)
                position, rotation = transform_2d_to_3d(p_position[:, observer_id], p_orient[observer_id])
                c_lidar = lidar_data[event_idx, :]
                if event_idx % UPDATE_INTERVAL == 0:
                    p_weight = particle_filter.update_particles(p_position, p_orient, p_weight, c_lidar, map, xm, ym)
                if event_idx % MAPPING_INTERVAL == 0:
                    map = mapping.update_map(map, xm, ym, rotation, position, c_lidar)
                p_position, p_orient, p_weight = particle_filter.resample_particles(p_position, p_orient, p_weight)
            
            elif event_type == "stereo":
                # Read RGB image
                path = os.path.join('data/stereo_left', "{}.png".format(stereo_time[event_idx]))
                img = cv2.imread(path, 0)
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_cp = np.zeros_like(img).reshape((-1, 3)).T.astype(np.float64)
                disparity_map = disparity[event_idx, :, :]
                camera_co = recover_space_coordinate(camera_transition, disparity_map)
                valid_idx = (disparity_map > 0).reshape(-1)
                target_co = camera_co.reshape((3, -1))[:, valid_idx]
                img = (img.reshape((-1, 3)).T)[:, valid_idx]
                # img_cp[:, valid_idx] = img
                observer_id = np.argmax(p_weight)
                position, rotation = transform_2d_to_3d(p_position[:, observer_id], p_orient[observer_id])
                st2wo_rotation = rotation @ STEREO_ROTATION
                st2wo_position = position + rotation @ STEREO_POSITION
                world_co = st2wo_position[:, None] + st2wo_rotation @ target_co
                img_cp[:, valid_idx] = world_co
                # plt.imshow(img_cp.reshape(3, 560, 1280)[1, :, :])
                # plt.show(block=True)
                # Constrain z idx
                zvalid_idx = np.logical_and((world_co[-1, :] < STEREO_Z_RANGE[1]), (world_co[-1, :] > STEREO_Z_RANGE[0]))
                img_cp[:, valid_idx] = img * zvalid_idx
                world_co = world_co[:-1, zvalid_idx]
                img = img[:, zvalid_idx]
                # print(np.sum(zvalid_idx) / zvalid_idx.size)
                # plt.imshow(img_cp.reshape(3, 560, 1280).transpose((1, 2, 0)))
                # plt.show(block=True)
                print("{} points detected!".format(world_co.shape[1]))
                # Add to color map
                xidx, yidx = physics2map(map, xm, ym, world_co[0, :], world_co[1, :])
                color_map[:, xidx, yidx] = img
                # Compute camera to world transform

        if t_idx % STEPS_TRAJECTORY == 0:
            xidx, yidx = physics2map(map, xm, ym, p_position[0, :], p_position[1, :])
            traj_x.append(xidx)
            traj_y.append(yidx)

        if t_idx and t_idx % STEPS_FIGURES == 0:
            plt.imshow(np.sign(map).T, origin='lower')
            x_trajs, y_trajs = np.vstack(traj_x), np.vstack(traj_y)
            plt.plot(x_trajs, y_trajs, color="red", linewidth=0.1)
            plt.axis("off")
            plt.savefig(os.path.join(IMG_OUTPUT_PATH, "step{}.png".format(t_idx)), dpi=600)
            plt.cla() 
            plt.clf() 
            plt.close('all')
            plt.imshow(color_map.T, origin='lower')
            plt.axis("off")
            plt.savefig(os.path.join(IMG_OUTPUT_PATH, "step_c{}.png".format(t_idx)), dpi=600)
            plt.cla() 
            plt.clf() 
            plt.close('all')

            gc.collect()
    plt.imshow(color_map.T, origin='lower')
    plt.axis("off")
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "colormap.jpg"), dpi=600)
    plt.cla() 
    plt.clf() 
    plt.close('all')

if __name__ == "__main__":
    main(NUM_PARTICLES)