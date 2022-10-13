#!/bin/bash

# Required: ffmpeg
# ffmpeg -framerate <int/float> -i <input-file-paths/names-pattern> -c:v libx264 -r 30 -pix_fmt yuv420p <output-file-name/path .mp4>
ffmpeg -framerate 0.8 -i fig_%02d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
# import subprocess
# # call the bash script as a subprocess using call() a blocking function
# subprocess.call("./scripts/images2mp4.sh")
