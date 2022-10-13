#!/bin/bash

# Required: ImageMagik
# convert -delay <delay-in-ms> -loop 0 <input-file-paths/names-pattern> <output-file-path/name>
convert -delay 120 -loop 0 "./results/figures/fig_*.png" "./results/videos/myGIF.gif"
# Use within a python script using:-
# import subprocess
# # call the bash script as a subprocess using call() a blocking function
# subprocess.call("./scripts/images2gif.sh")
