#!/bin/bash

# this script flips, flops, and rotates our labeled data to ~augment~ it

# replace with your path
cd /Users/dklink/data_science/rubbish_net/labeled_data/trash_images
mkdir -p augmented_images  # create if doesn't already exist

for f in *.JPG; do
  convert -flip "$f" "augmented_images/${f%.JPG}.flip.JPG";
  convert -flop "$f" "augmented_images/${f%.JPG}.flop.JPG";
  convert -rotate 0 "$f" "augmented_images/${f%.JPG}.rotate0.JPG";
  convert -rotate 90 "$f" "augmented_images/${f%.JPG}.rotate90.JPG";
  convert -rotate 180 "$f" "augmented_images/${f%.JPG}.rotate180.JPG";
  convert -rotate 270 "$f" "augmented_images/${f%.JPG}.rotate270.JPG";
done
