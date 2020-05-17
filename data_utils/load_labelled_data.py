import pandas as pd
import os

LABELS_PATH = '../labeled_data/'


def load_labelbox_json():
    data = pd.read_json(os.path.join(LABELS_PATH, 'export-2020-05-14T18_46_40.167Z.json'))
    # construct a dataframe:  columns: picture_title, bboxes

    def extract_bboxes(row):
        label = row['Label']
        if not label:
            return []

        objects = label['objects']
        bboxes = []
        for o in objects:
            bboxes.append(o['bbox'])
        return bboxes

    labels = pd.DataFrame(columns=['filename', 'bboxes'])
    labels['filename'] = data['External ID']
    labels['bboxes'] = data.apply(extract_bboxes, axis=1)

    return labels
