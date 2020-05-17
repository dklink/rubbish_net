import pandas as pd

from data_utils.crop_bboxes import crop_from_labels
from data_utils.load_labeled_data import load_labelbox_json

IN_DIR = '/Volumes/Seagate\\ Backup+\\ P/robindevries-35c328/10.01/cropped/'
OUT_DIR = '../labeled_data/trash_images/'


def generate_positives():
    labels = load_labelbox_json()
    # randomly drop one of the assets which was labeled twice (once by Doug, once by Sean)
    labels = labels.sample(frac=1).drop_duplicates(subset='filename')

    filenames = []
    bboxes = []
    for row in labels.itertuples():
        for bbox in row.bboxes:
            filenames.append(row.filename)
            bboxes.append(bbox)
    crop_from_labels(pd.DataFrame({'filename': filenames, 'bbox': bboxes}), IN_DIR, OUT_DIR)

generate_positives()
