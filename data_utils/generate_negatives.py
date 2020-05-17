from data_utils.crop_bboxes import crop_from_labels
from data_utils.load_labeled_data import load_labelbox_json
import random
import pandas as pd
import numpy as np

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

    #crop_from_labels(negative_labels, IN_DIR, OUT_DIR)
    return negative_labels


def is_valid(negative_bbox, positive_bboxes):
    """boolean, determines whether the generated negative_bbox is a valid bounding box.
    Criteria for validity
        1. The negative bounding box overlaps with none of the positive bounding boxes
        2. ??? any others you can think of
    A bounding box is a dictionary with keys ['width', 'height', 'top', 'left']
    :param negative_bbox: bounding box to test
    :param positive_bboxes: list of bounding boxes against which to test negative_bbox
    """
    if len(positive_bboxes) == 0:
        return False

    pos = np.empty((len(positive_bboxes), 4))
    for i in range(len(positive_bboxes)):
        pos[i, 0] = positive_bboxes[i]["width"]
        pos[i, 1] = positive_bboxes[i]["height"]
        pos[i, 2] = positive_bboxes[i]["left"]
        pos[i, 3] = positive_bboxes[i]["top"]

    x_min = negative_bbox["left"]
    x_max = negative_bbox["left"] + negative_bbox["width"]
    y_min = negative_bbox["top"]
    y_max = negative_bbox["top"] + negative_bbox["height"]

    pos_x_mins = pos[:, 2]
    pos_x_maxs = pos[:, 2] + pos[:, 0]
    pos_y_mins = pos[:, 3]
    pos_y_maxs = pos[:, 3] + pos[:, 1]

    x_good = np.all(pos_x_mins >= x_max or pos_x_maxs <= x_min)
    y_good = np.all(pos_y_mins >= y_max or pos_y_maxs <= y_min)
    return x_good and y_good


def test_is_valid():
    positive_bboxes = []
    positive_bboxes.append({"top": 1, "left": 0, "height": 1, "width": 1})
    negative_bbox = {"top": 2, "left": 0, "height": 2, "width": 2}
    print(is_valid(negative_bbox, positive_bboxes))
