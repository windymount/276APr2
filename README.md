# Particle filter SLAM
This project used data collected from sensors and stereo camera on an autonomous car to perform simultaneously localization and mapping (SLAM).
## Quick start
1. Create directories for saving data and output images.
```
mkdir img data
```
2. Download the data and put them under `data\` with following structure
```
Particle_filter_SLAM
└───data
    │
    └───param
    │
    └───sensor_data
    │
    └───stereo_left
    │
    └───stereo_right
```
3. Run the script
```
python run_slam.py
```

## Demo

https://user-images.githubusercontent.com/44419662/187271535-44d5d327-fa26-4b4e-9f87-1727f00f218d.mp4

https://user-images.githubusercontent.com/44419662/187271563-15a96fd3-70ae-464e-ba77-5070f96d462f.mp4

