import numpy as np

def generate_negatives():
    """randomly generate images of NOT plastic from our dataset"""


def is_valid(negative_bbox, positive_bboxes):
    """boolean, determines whether the generated negative_bbox is a valid bounding box.
    Criteria for validity
        1. The negative bounding box overlaps with none of the positive bounding boxes
        2. ??? any others you can think of
    A bounding box is a list of 4 integers, [width, height, left, top]
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
