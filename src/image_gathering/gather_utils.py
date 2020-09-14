import cv2
import io
from multiprocessing.pool import ThreadPool
import numpy as np
import os
import pandas as pd
import requests

def load_data():
    """Load train/test WEBEmo URLs and labels."""
    train = pd.read_csv(r'../data/csvs/train25.txt',
                        sep=" ",
                        header=None).rename(columns={0:'url', 1:'label'})
    test = pd.read_csv(r'../data/csvs/test25.txt',
                       sep=" ",
                       header=None).rename(columns={0:'url', 1:'label'})

    train['file_label'] = train['label'].map(lambda x: str(x)) + '_' + train['url'].map(lambda x: x.split('/')[-1])
    test['file_label'] = test['label'].map(lambda x: str(x)) + '_' + test['url'].map(lambda x: x.split('/')[-1])

    train['root_dir'] = '../data/images/train'
    test['root_dir'] = '../data/images/test'
    return train, test


def save_imgs(df):
    """Retrieves an image from its URL and saves locally.

    Args:
        df (dataframe): Pandas dataframe containing image
          URLs and associated labels.

    Returns:
        None. Image is saved locally.
    """
    response = requests.get(df[1]['url'])
    img = Image.open(BytesIO(response.content))
    img_name = df[1]['url'].split('/')[-1]
    label = str(df[1]['label'])
    path = os.path.join(df[1]['root_dir'], label+'_'+img_name)
    img.save(path)
    return

def pool_image_retrieval(df):
    """Utilizes multithreading to call `save_imgs()` function.

    Args:
        df (dataframe): Pandas dataframe containing image
          URLs and associated labels.

    Returns:
        None. Images are saved locally.
    """
    pool = ThreadPool()
    pool.map(func=save_imgs, iterable=df.iterrows())
    return

def label_dicts():
    """Returns dictionaries mapping level-3 labels to levels 1 and 2, respectively."""

    # Index of labels corresponds to label column in
    # dataframe e.g., 'affection' is '0' in train/test
    labels=[('affection', 'love', '+'),
            ('cheerfullness', 'joy', '+'),
            ('confusion', 'sadness', '-'),
            ('contentment', 'joy', '+'),
            ('disappointment', 'sadness', '-'),
            ('disgust', 'anger', '-'),
            ('enthrallment', 'joy', '+'),
            ('envy', 'anger', '-'),
            ('exasperation', 'anger', '-'),
            ('gratitude', 'love', '+'),
            ('horror', 'fear', '-'),
            ('irritabilty', 'anger', '-'),
            ('lust', 'love', '+'),
            ('neglect', 'sadness', '-'),
            ('nervousness', 'fear', '-'),
            ('optimism', 'joy', '+'),
            ('pride', 'joy', '+'),
            ('rage', 'anger', '-'),
            ('relief', 'joy', '+'),
            ('sadness', 'sadness', '-'),
            ('shame', 'sadness', '-'),
            ('suffering', 'sadness', '-'),
            ('surprise', 'surprise', '+'),
            ('sympathy', 'sadness', '-'),
            ('zest', 'joy', '+')]

    lvl_one = {}
    lvl_two = {}
    for idx, val in enumerate(labels):
        lvl_one[idx] = val[2]
        lvl_two[idx] = val[1]

    return lvl_one, lvl_two

def retrieve_labels(train, test, l1_map, l2_map):
    """Numerically encodes level-1 and level-2 WEBEmo labels.

    Args:
        train, test (DataFrame): Pandas dataframe containing
          'lvl_three' series.
        l1_map, l2_map (dict): Dictionary containing level-3
          codes as keys and level-1/level-2 labels as values.

    Returns:
        Pandas dataframe with 'lvl_one' and 'lvl_two' series
        added.
    """
    for df in [train, test]:
        df['lvl_one'] = df.replace({'lvl_three':l1_map})['lvl_three']
        df['lvl_two'] = df.replace({'lvl_three':l2_map})['lvl_three']

    l1_dict = {v:k for k,v in dict(enumerate(train['lvl_one'].unique())).items()}
    l2_dict = {v:k for k,v in dict(enumerate(train['lvl_two'].unique())).items()}

    for df in [train, test]:
        df['lvl_one'] = df['lvl_one'].replace(l1_dict)
        df['lvl_two'] = df['lvl_two'].replace(l2_dict)

    return train, test

