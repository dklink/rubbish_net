from load_labelled_data import load_labelbox_json
import multiprocessing as mp
import os
IN_DIR = '/Volumes/Seagate\\ Backup+\\ P/robindevries-35c328/10.01/cropped/'
OUT_DIR = 'labeled_trash/'

labels = load_labelbox_json()
# randomly drop one of the assets which was labeled twice (once by Doug, once by Sean)
labels = labels.sample(frac=1).drop_duplicates(subset='filename')

commands = []
for row in labels.itertuples():
    in_path = IN_DIR + row.filename
    for i, bbox in enumerate(row.bboxes):
        out_path = OUT_DIR + f'bbox_{i}.' + row.filename
        commands.append(f"convert -crop {bbox['width']}x{bbox['height']}+{bbox['left']}+{bbox['top']} {in_path} {out_path}")


def execute(idx):
    print(commands[idx])
    os.system(commands[idx])


with mp.Pool(processes=8) as pool:
    pool.map(execute, range(len(commands)))
