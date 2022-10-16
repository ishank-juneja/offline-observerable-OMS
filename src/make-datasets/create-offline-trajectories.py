"""
Make simple model datasets for training encoders
"""
# Some of these import are needed even they haven't been used explicitly here
import argparse
from arm_pytorch_utilities.rand import seed
import glob
import gym
# pycharm may not highlight this one but it is needed
import gym_cenvs
import numpy as np
import os
from src.utils import EncDataset
from src.config import SegConfig, CommonEncConfig
from src.simp_mod_datasets import FramesHandler, nsd, SimpleModel
from src.plotting import GIFMaker, SMVOffline


def main(args):



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder",
                        action='store',
                        type=str,
                        help="Name of the folder into which to put dataset, "
                             "should follow format of EncFolder even if not saving generated frames to disk",
                        metavar="folder")

    args = parser.parse_args()

    main(args)
