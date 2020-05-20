from copy import copy

import pandas as pd

from data_utils.crop_bboxes import crop_and_resize_from_labels
from data_utils.load_labeled_data import load_labelbox_json

IN_DIR = '/Volumes/SeagateBackup+P/robindevries-35c328/10.01/cropped/'
OUT_DIR = '../labeled_data/trash/'

IMAGE_HEIGHT = 1728
IMAGE_WIDTH = 3888


def generate_positives():
    labels = load_labelbox_json()
    # randomly drop one of the assets which was labeled twice (once by Doug, once by Sean)
    labels = labels.sample(frac=1).drop_duplicates(subset='filename')

    filenames = []
    bboxes = []
    failures = 0
    successes = 0
    for row in labels.itertuples():
        for bbox in row.bboxes:
            bbox = expand_bbox_to_square(bbox)
            # check bbox still fits in image
            if fits_in_image(bbox, IMAGE_WIDTH, IMAGE_HEIGHT):
                filenames.append(row.filename)
                bboxes.append(bbox)
                successes += 1
            else:
                failures += 1
    print(f"bbox expansion failures: {failures}, {failures/(failures+successes):.0%} of bboxes")
    crop_and_resize_from_labels(pd.DataFrame({'filename': filenames, 'bbox': bboxes}), 64, IN_DIR, OUT_DIR)


def fits_in_image(bbox, image_width, image_height):
    return (bbox['left'] >= 0 and bbox['left'] + bbox['width'] <= image_width and
            bbox['top'] >= 0 and bbox['top'] + bbox['height'] <= image_height)


def expand_bbox_to_square(in_bbox):
    bbox = copy(in_bbox)
    # expand bbox to square
    if bbox['width'] > bbox['height']:
        bbox['top'] -= (bbox['width'] - bbox['height']) // 2
        bbox['height'] = bbox['width']
    elif bbox['width'] < bbox['height']:
        bbox['left'] -= (bbox['height'] - bbox['width']) // 2
        bbox['width'] = bbox['height']
    return bbox

