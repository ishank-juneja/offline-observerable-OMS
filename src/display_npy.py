"""
Open .npy files by selecting them using a browse GUI
"""
import numpy as np
import tkinter
from tkinter import filedialog
import os
import argparse
from matplotlib import pyplot as plt

# Add argument for displaying as image via matpltolib
parser = argparse.ArgumentParser()
# Pass name of environment from within file gym_cenvs __init__.py
parser.add_argument("--image", action="store_true", default="ContinuousCartpole-v0")
args = parser.parse_args()

root = tkinter.Tk()
# use to hide tkinter window
root.withdraw()
currdir = os.getcwd()
# Retrieve path of np_files via browser GUI
npy_paths = filedialog.askopenfilenames(parent=root, initialdir=currdir, title='Please select a directory')
# Set options courtesy: https://stackoverflow.com/a/21529105/3642162
np.set_printoptions(threshold=1000, edgeitems=10, linewidth=140, formatter=dict(float=lambda x: "%.3g" % x))
for path in npy_paths:
    my_arr = np.load(path)
    # Print the file name
    file_name = path.split('/')[-1]
    print(file_name)
    # Set output print precision
    print(my_arr)
    # Attempt to diaplty as image
    if args.image:
        try:
            plt.imshow(my_arr)
            plt.show()
        except:
            print("Couldn't display array with matplotlib, shape was {0}".format(my_arr.shape))
