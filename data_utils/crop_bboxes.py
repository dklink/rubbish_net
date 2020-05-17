import os
import multiprocessing as mp


def execute_command(command):
    print(command + '\n', end='')
    os.system(command)


def crop_from_labels(labels, in_dir, out_dir):
    """
    :param labels: pandas Dataframe with columns 'filename', 'bbox'
    :param in_dir: location of files referenced in 'labels'
    :param out_dir: location to put cropped images
    """
    commands = []
    for row in labels.itertuples():
        in_path = os.path.join(in_dir, row.filename)
        crop_dims = f"{row.bbox['width']}x{row.bbox['height']}+{row.bbox['left']}+{row.bbox['top']}"
        out_path = os.path.join(out_dir, crop_dims + "_" + row.filename)
        commands.append(f"convert -crop {row.bbox['width']}x{row.bbox['height']}+{row.bbox['left']}+{row.bbox['top']} {in_path} {out_path}")

    with mp.Pool(processes=8) as pool:
        pool.map(execute_command, commands)
