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

