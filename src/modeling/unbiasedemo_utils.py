import os
import pandas

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
                file_list.append(str('/'.join(path.split('/')[-2:])+'/'+name))

    labels = ['surprise', 'fear', 'love', 'anger', 'joy', 'sadness']
    label_dict = {}
    for idx, val in enumerate(labels):
        label_dict[val] = idx

    unbiasedemo_df = pd.DataFrame({'file':file_list, 'labels': [f.split('/')[0] for f in file_list]})
    unbiasedemo_df['labels'] = unbiasedemo_df['labels'].replace(label_dict)

    return unbiasedemo_df
