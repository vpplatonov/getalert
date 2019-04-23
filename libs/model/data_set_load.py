# https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
import numpy as np
import pandas as pd

import os
import argparse

from tqdm import tqdm
from predict.feature_engineer import get_mfcc, NUM_MFCC, SAMPLE_RATE, NUM_PCA, extract_features

tqdm.pandas()

FOLDS = [1, 3, 4]
TARGETS = []

# For ESC-50 dataset2019curated
# PATH_SUFFIX = '../ESC-50/'
# PATH_SUFFIX_SAVE = '../'
#
# FNAME_COLUMN = 'filename'
# LNAME_COLUMN = 'category'
# TARGETS = [0,1,2,3,4,5,6,7,8,9,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
#
# csv_file = 'esc50'
# target_folder = 'audio_origin'
# meta_folder = 'meta/'

# For freesound-audio_origin-tagging-2019
PATH_SUFFIX = '../freesound-audio_origin-tagging-2019/'
PATH_SUFFIX_SAVE = '../'
csv_file = 'train_curated'
target_folder = 'train_curated'
FNAME_COLUMN = 'fname'
LNAME_COLUMN = 'labels'
meta_folder = ''


def filter_exeptions(data_extra, only_clear=True):
    """
    for freesound-audio_origin-tagging-2019
    Get only clear sound tagged by only one class
    :param data_extra:
    :return:
    """
    data_extra_clear = data_extra[~data_extra[LNAME_COLUMN].str.contains(",")]
    if only_clear:
        return data_extra_clear
    else:
        labels = data_extra_clear[LNAME_COLUMN].unique().tolist()
        labels_regexp = '|'.join(labels).replace('(', '\(').replace(')', '\)')
        data_extra_combined_only = data_extra[~data_extra[LNAME_COLUMN].str.contains(labels_regexp)]

        return data_extra_combined_only


def data_set_load():
    audio_train_files = os.listdir(f"../{PATH_SUFFIX}{target_folder}")
    # audio_test_files = os.listdir(f"../{PATH_SUFFIX}audio_test")

    train = pd.read_csv(f"../{PATH_SUFFIX}{meta_folder}{csv_file}.csv")
    # filter
    if len(TARGETS) > 0:
        train = train[train['target'].isin(TARGETS)]
    else:
        train = filter_exeptions(train)

    return train, audio_train_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='{}/../{}output/dataset/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_SAVE
                        ))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)

    train, audio_train_files = data_set_load()

    # Prepare data
    train_data = pd.DataFrame()
    train_data[FNAME_COLUMN] = train[FNAME_COLUMN]

    train_data = train_data[FNAME_COLUMN].progress_apply(get_mfcc, path=f"../{PATH_SUFFIX}{target_folder}/")
    print('done loading train mfcc')
    train_data[FNAME_COLUMN] = train[FNAME_COLUMN]
    train_data[LNAME_COLUMN] = train[LNAME_COLUMN]

    # https://www.kaggle.com/tetyanayatsenko/xgb-using-mfcc-opanichev-s-featur-02
    train_files = train_data[FNAME_COLUMN].values
    train_features = extract_features(train_files,  f"../{PATH_SUFFIX}{target_folder}", fname=FNAME_COLUMN)
    train_data = train_data.merge(train_features, on=FNAME_COLUMN, how='left')
    print(train_data.shape)

    # Construct features set
    X = train_data.drop([LNAME_COLUMN, FNAME_COLUMN], axis=1)
    X = X.values
    labels = np.sort(np.unique(train_data[LNAME_COLUMN].values))
    num_class = len(labels)

    print(f"Feature names {', '.join(labels)}")
    print(f"Class nums {num_class}")

    c2i = {}
    i2c = {}
    for i, c in enumerate(labels):
        c2i[c] = i
        i2c[i] = c
    y = np.array([c2i[x] for x in train_data[LNAME_COLUMN].values])

    # Save features
    # Save to numpy binary format
    np.save(os.path.join(load_path, 'dataset.npy'), X, fix_imports=False)
    np.save(os.path.join(load_path, 'labels.npy'), y)
    np.save(os.path.join(load_path, 'to_labels.npy'), i2c, fix_imports=False)


if __name__ == '__main__':
    main()
