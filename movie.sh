#! /usr/bin/env bash

# Make a movie demonstrating the behaviour of the optical component across a range of angles
# of incidence.

# TODO: Use geometric component to show zenith/azimuth angle of the sun at the given time.
# TODO: The spectrum might change over the observing window, can we generate more spectra using a
# PSG API?

# Clear any existing frames
[ -d frames ] && rm -r frames && mkdir frames

# Run python script to generate movie frames
# TODO: What information should be passed into the Python script?
# TODO: Calculate no. frames, duration, animation metadata to pass into Python script
title=geometry
python "$title.py"

# Stitch movie frames together with ffmpeg
ffmpeg -y -framerate 30 -pattern_type glob -i 'frames/*.png' -c:v libx264 -pix_fmt yuv420p "$title.mp4"
# Create GIF from movie
ffmpeg -y -i "$title.mp4" "$title.gif"

# Play movie with xdg-open
xdg-open "$title.mp4"



