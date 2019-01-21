# https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
import numpy as np
import pandas as pd

import os
import argparse

from tqdm import tqdm
from model.feature_engineer import get_mfcc, NUM_MFCC, SAMPLE_RATE

tqdm.pandas()

PATH_SUFFIX = '../ESC-50-master/'
PATH_SUFFIX_SAVE = '../'

FNAME_COLUMN = 'filename'
LNAME_COLUMN = 'category'
FOLDS = [1, 3, 4]
TARGETS = [0,1,2,3,4,5,6,7,8,9,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
NUM_PCA = 65


def convert_to_labels(preds, i2c, k=2):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids


def data_set_load():
    audio_train_files = os.listdir(f"../{PATH_SUFFIX}audio")
    # audio_test_files = os.listdir(f"../{PATH_SUFFIX}audio_test")

    train = pd.read_csv(f"../{PATH_SUFFIX}meta/esc50.csv")
    # filter
    train = train[train['target'].isin(TARGETS)]
    return train, audio_train_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='{}/../{}output/dataset/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_SAVE
                        ))
    parser.add_argument('--save_path',
                        default='{}/../{}output/model/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_SAVE
                        ))
    parser.add_argument('--log_path',
                        default='{}/../../ESC-50-master/'.format(
                            os.path.dirname(os.path.abspath(__file__))
                        ))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    save_path = os.path.normpath(args.save_path)
    log_path = os.path.normpath(args.log_path)

    train, audio_train_files = data_set_load()
    # Prepare data
    train_data = pd.DataFrame()
    train_data[FNAME_COLUMN] = train[FNAME_COLUMN]

    train_data = train_data[FNAME_COLUMN].progress_apply(get_mfcc, path=f"../{PATH_SUFFIX}audio/")
    print('done loading train mfcc')

    train_data[FNAME_COLUMN] = train[FNAME_COLUMN]
    train_data[LNAME_COLUMN] = train[LNAME_COLUMN]

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
    # logging.info('Saving training set...')
    np.save(os.path.join(load_path, 'dataset.npy'), X, fix_imports=False)
    np.save(os.path.join(load_path, 'labels.npy'), y)
    np.save(os.path.join(load_path, 'to_labels.npy'), i2c, fix_imports=False)


if __name__ == '__main__':
    main()
