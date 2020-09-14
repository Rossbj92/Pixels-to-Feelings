import os
import pandas as pd

def make_df(path):
    """Constructs a dataframe containing image names and labels.

    Args:
        Path (str): Path to directory where images are located.

    Returns:
        Dataframe containing image file paths and labels.
    """
    file_list = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if '.jpg' in name:
                file_list.append(str(path.split('/')[-1])+'/'+name)
    file_list = [f for f in file_list if '.ipynb' not in f]

    labels = set([f.split('/')[0] for f in file_list])
    label_dict = {}
    for idx, val in enumerate(labels):
        label_dict[val] = idx

    df = pd.DataFrame({'file':file_list, 'labels': [f.split('/')[0] for f in file_list]})
    df['labels'] = df['labels'].replace(label_dict)
    return df
