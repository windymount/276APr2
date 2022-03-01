"""

Run the particle filter SLAM algorithm.

"""
from pr2_utils import compute_stereo
from particleSLAM import main
import params
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n-particle", type=int, default=params.NUM_PARTICLES, help="Number of particles")
parser.add_argument("--output", type=str, default=params.IMG_OUTPUT_PATH, help="Output path of images")
args = parser.parse_args()
# Precompute stereo cameras disparity
compute_stereo()
main(args.n_particle, args.output)
