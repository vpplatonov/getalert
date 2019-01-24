
import numpy as np
import pandas as pd

from scipy.stats import skew

import librosa

PATH_SUFFIX_LOAD = '../'
# PATH_SUFFIX_LOAD = '../ESC-50-master/'
# PATH_SUFFIX_SAVE = '../ESC-50-master/'
PATH_SUFFIX_SAVE = '../'
SAMPLE_RATE = 16000
NUM_MFCC = 30
FRAME = 512
NUM_PCA = 65


def convert_to_labels(preds, i2c, k=2):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids


def get_mfcc_data(name, path):
    data, _ = librosa.core.load(path + name,
                                sr=SAMPLE_RATE)
    assert _ == SAMPLE_RATE

    return data


def get_mfcc_feature(data):
    """ Generate mfcc features with mean and standard deviation
            all librosa features have hop_length=512 by default
        """

    try:
        ft1 = librosa.feature.mfcc(data, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
        ft2 = librosa.feature.zero_crossing_rate(data)[0]
        ft3 = librosa.feature.spectral_rolloff(data)[0]
        ft4 = librosa.feature.spectral_centroid(data)[0]
        ft5 = librosa.feature.spectral_contrast(data)[0]
        ft6 = librosa.feature.spectral_bandwidth(data)[0]
        ft1_trunc = np.hstack((np.mean(ft1, axis=1),
                               np.std(ft1, axis=1),
                               skew(ft1, axis=1),
                               np.max(ft1, axis=1),
                               np.median(ft1, axis=1),
                               np.min(ft1, axis=1)))
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
        ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
        ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
        return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))
    except:
        print('bad file')
        return pd.Series([0] * 210)


def get_mfcc(name, path):
    """ Used for train model """
    data = get_mfcc_data(name, path)

    return get_mfcc_feature(data)
