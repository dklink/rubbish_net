from data_utils.crop_bboxes import crop_from_labels
from data_utils.load_labeled_data import load_labelbox_json
import random
import pandas as pd

IN_DIR = '/Volumes/Seagate\\ Backup+\\ P/robindevries-35c328/10.01/cropped/'
OUT_DIR = '../labeled_data/not_trash_images/'


IMAGE_HEIGHT = 1728
IMAGE_WIDTH = 3888


def generate_negatives(num_negatives):
    """randomly generate images of NOT plastic from our dataset"""
    # load positive bounding boxes and filenames
    positive_labels = load_labelbox_json()
    positive_bboxes = []
    for row in positive_labels.itertuples():
        positive_bboxes += row.bboxes
    filenames = positive_labels['filename'].to_list()

    # generate negative candidates
    negative_bboxes = []
    while len(negative_bboxes) < num_negatives:
        negative_bbox = random.choice(positive_bboxes)
        # retain width/height, but randomly choose an image,
        # and randomly choose a new 'top' and 'left', making sure resultant box fits in image
        negative_bbox['left'] = random.randint(0, IMAGE_WIDTH-negative_bbox['width'])
        negative_bbox['top'] = random.randint(0, IMAGE_HEIGHT-negative_bbox['height'])

        # check doesn't overlap with any plastic boxes
        if is_valid(negative_bbox, positive_bboxes):
            negative_bboxes.append(negative_bbox)

    # randomly generate files to crop (with replacement)
    negative_filenames = random.choices(filenames, k=num_negatives)
    negative_labels = pd.DataFrame({'filename': negative_filenames, 'bboxes': negative_bboxes})

    crop_from_labels(negative_labels, IN_DIR, OUT_DIR)


def is_valid(negative_bbox, positive_bboxes):
    """boolean, determines whether the generated negative_bbox is a valid bounding box.
    Criteria for validity
        1. The negative bounding box overlaps with none of the positive bounding boxes
        2. ??? any others you can think of
    A bounding box is a dictionary with keys ['width', 'height', 'top', 'left']
    :param negative_bbox: bounding box to test
    :param positive_bboxes: list of bounding boxes against which to test negative_bbox
    """

    # YOUR CODE HERE :)

    return True
