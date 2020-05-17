#!/bin/bash

# this script crops our webcam images from full size to be the bottom left section of the image.

# replace with your path
cd /Volumes/Seagate\ Backup+\ P/robindevries-35c328/10.01/
mkdir -p cropped  # create if doesn't already exist
# each image is 5184x3456
# we cut in half vertical, and in 3/4 horizontal
# put all of the cropped images in the cropped folder
mogrify -crop 3888x1728+0+1728 -monitor -path ./cropped ./*.JPG

# insert "cropped" into their names
cd cropped
for f in *.JPG;
  do mv "$f" "${f%.JPG}.cropped.JPG";
done