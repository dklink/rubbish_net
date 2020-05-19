from data_utils.crop_bboxes import crop_and_resize_from_labels
from data_utils.generate_positives import fits_in_image, expand_bbox_to_square
from data_utils.load_labeled_data import load_labelbox_json

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib import patches

IN_DIR = '/Volumes/SeagateBackup+P/robindevries-35c328/10.01/cropped/'
OUT_DIR = '../labeled_data/not_trash_images/'

IMAGE_HEIGHT = 1728
IMAGE_WIDTH = 3888


def generate_negatives(num_negatives, out_size=64):
    """randomly generate images of NOT plastic from our dataset"""
    # load positive bounding boxes and filenames
    positive_labels = load_labelbox_json()
    positive_bboxes = []
    positive_filenames = []
    for row in positive_labels.itertuples():
        for bbox in row.bboxes:
            positive_bboxes.append(bbox)
            positive_filenames.append(row.filename)

    # generate random filenames for each random bbox
    candidate_filenames = random.choices(positive_filenames, k=num_negatives)

    # generate negative candidates
    candidate_bboxes = []
    candidate_idx = 0
    while candidate_idx < num_negatives:
        negative_bbox = expand_bbox_to_square(random.choice(positive_bboxes))  # choose a bbox and make it square
        # retain width/height and randomly choose a new 'top' and 'left', making sure resultant box fits in image
        negative_bbox['left'] = random.randint(0, IMAGE_WIDTH-negative_bbox['width'])
        negative_bbox['top'] = random.randint(0, IMAGE_HEIGHT-negative_bbox['height'])

        # check doesn't overlap with any bboxes in same image
        this_image_bboxes = positive_labels[positive_labels.filename == candidate_filenames[candidate_idx]].bboxes.iloc[0]
        if fits_in_image(negative_bbox, IMAGE_WIDTH, IMAGE_HEIGHT) and no_overlaps(negative_bbox, this_image_bboxes):
            candidate_bboxes.append(negative_bbox)
            candidate_idx += 1
        # else:
        #    print(negative_bbox, this_image_bboxes)
        #    show_bboxes([negative_bbox] + this_image_bboxes, candidate_filenames[candidate_idx])

    # randomly generate files to crop (with replacement)
    negative_labels = pd.DataFrame({'filename': candidate_filenames, 'bbox': candidate_bboxes})
    crop_and_resize_from_labels(negative_labels, out_size, IN_DIR, OUT_DIR)
    return negative_labels


def no_overlaps(negative_bbox, positive_bboxes):
    """boolean, determines whether the generated negative_bbox overlaps with none of the positive bounding boxes
    A bounding box is a dictionary with keys ['width', 'height', 'top', 'left']
    :param negative_bbox: bounding box to test
    :param positive_bboxes: list of bounding boxes against which to test negative_bbox
    """
    if len(positive_bboxes) == 0:
        return True

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

    x_good = np.logical_or(pos_x_mins >= x_max, pos_x_maxs <= x_min)
    y_good = np.logical_or(pos_y_mins >= y_max, pos_y_maxs <= y_min)
    return np.all(np.logical_or(x_good, y_good))


def test_no_overlaps():
    positive_bboxes = [{"top": 1, "left": 0, "height": 1, "width": 1},
                       {"top": -1, "left": -1, "height": 0.5, "width": 0.5}]
    negative_bbox = {"top": 2, "left": 0, "height": 2, "width": 2}
    print(no_overlaps(negative_bbox, positive_bboxes))


def show_bboxes(bboxes, filename):
    """shows the first bbox in black"""
    img = image.imread(IN_DIR + filename)
    figure, ax = plt.subplots(1)
    ax.imshow(img)
    color = "yellow"
    for bbox in bboxes:
        rect = patches.Rectangle((bbox['left'], bbox['top']), bbox['width'], bbox['height'],
                                 edgecolor=color, facecolor="none")
        color = "orange"
        ax.add_patch(rect)


#generate_negatives(800)
