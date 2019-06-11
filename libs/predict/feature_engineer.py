
import numpy as np
import pandas as pd

from scipy.stats import skew
from tqdm import tqdm, tqdm_pandas
import librosa
import os
import scipy
import math
from pydub import AudioSegment


tqdm.pandas()

PATH_SUFFIX_LOAD = '../'
# PATH_SUFFIX_LOAD = '../ESC-50-master/'
# PATH_SUFFIX_SAVE = '../ESC-50-master/'
PATH_SUFFIX_SAVE = '../'
SAMPLE_RATE = 16000
SOUND_DURATION = 3
NUM_MFCC = 30
FRAME = int(SAMPLE_RATE / 1000 * 10)  # 10 ms
NUM_PCA = 65
MODEL_TYPE = 'XGBoost'  # 'SVC'
FOLDER = 'XGBoost3'


def conf_load(dataroot, folder=FOLDER):
    # print(str(dataroot / folder / 'conf.npy'))
    if os.path.exists(str(dataroot / folder / 'conf.npy')):
        conf = np.load(dataroot / folder / 'conf.npy').tolist()
    else:
        conf = np.array([])
        conf['sampling_rate'] = SAMPLE_RATE
        conf['duration'] = SOUND_DURATION
        conf['n_mfcc'] = NUM_MFCC
        conf['hop_length'] = FRAME

    conf['learning_rate'] = 0.0001
    conf['samples'] = conf['sampling_rate'] * conf['duration']
    conf['dims'] = (conf['n_mels'], 1 + int(np.floor(conf['samples'] / conf['hop_length'])), 1)
    conf['normalize'] = 'featurewise'
    conf['folder'] = FOLDER

    return conf


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


def read_audio(conf, pathname):
    y, sr = librosa.load(str(pathname), sr=conf['sampling_rate'])
    # trim silence
    if 0 < len(y):  # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)

    # make it unified length to conf.samples
    if len(y) > conf['samples']:  # long enough
        if conf['audio_split'] == 'head':
            y = y[0:0 + conf['samples']]
    else:  # pad blank
        # print('pad blank')
        padding = conf['samples'] - len(y)  # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf['samples'] - len(y) - offset), 'constant')
    return y


def audio_load(conf, pathname, pydub_read=False):
    play_list = list()

    # load audio with different length
    # https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data from Data Generator
    # Read and Resample the audio
    if not pydub_read:
        data, _ = librosa.core.load(str(pathname), sr=conf['sampling_rate']
                                    # res_type='kaiser_fast'
                                    )
    else:
        main_config = {
            'sampling_rate': conf['sampling_rate'],
            'channels': 1,
            'sample_width': 2  # bit rate 16 bit
        }

        def file_to_array(wav_path, cnf):
            # read wav file
            inp_audio = AudioSegment.from_file(wav_path, "wav")
            # set the sample rate
            inp_audio = inp_audio.set_frame_rate(cnf['sampling_rate'])
            # set the numbers of channels
            inp_audio = inp_audio.set_channels(cnf['channels'])
            # convert wav to 16bit
            inp_audio = inp_audio.set_sample_width(cnf['sample_width'])
            # convert to vector array and return
            samples = inp_audio.get_array_of_samples()

            return np.array([(s / 2 ** 16.0) * 2 for s in samples])
            # return np.array(inp_audio.get_array_of_samples())

        data = file_to_array(str(pathname), main_config)

    input_length = conf['samples']

    # Random offset / Padding
    if len(data) > conf['samples']:

        for offset in range(math.floor((len(data) - input_length) / conf['sampling_rate']) + 1):
            # Feature extraction
            tmp = get_mfcc_feature(data[(offset * conf['sampling_rate']):int(offset * conf['sampling_rate'] + input_length)],
                                   conf)
            play_list.append(tmp)
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, int(input_length - len(data) - offset)), "constant")
        # Feature extraction
        tmp = get_mfcc_feature(data, conf)
        play_list.append(tmp)

    return play_list


def get_mfcc_feature(data, conf=None):
    """ Generate mfcc features with mean and standard deviation
        all librosa features have hop_length=512 by default
    """

    try:
        ft1 = librosa.feature.mfcc(data,
                                   sr=SAMPLE_RATE if conf is None else conf['sampling_rate'],
                                   n_mfcc=NUM_MFCC if conf is None else conf['n_mfcc'])
        ft2 = librosa.feature.zero_crossing_rate(data,
                                                 hop_length=FRAME if conf is None else conf['hop_length'])[0]
        ft3 = librosa.feature.spectral_rolloff(data,
                                               sr=SAMPLE_RATE if conf is None else conf['sampling_rate'],
                                               hop_length=FRAME if conf is None else conf['hop_length'])[0]
        ft4 = librosa.feature.spectral_centroid(data,
                                                sr=SAMPLE_RATE if conf is None else conf['sampling_rate'],
                                                hop_length=FRAME if conf is None else conf['hop_length'])[0]
        ft5 = librosa.feature.spectral_contrast(data,
                                                sr=SAMPLE_RATE if conf is None else conf['sampling_rate'],
                                                n_bands=6,
                                                fmin=200.0)[0]
        ft6 = librosa.feature.spectral_bandwidth(data,
                                                 sr=SAMPLE_RATE if conf is None else conf['sampling_rate'],
                                                 hop_length=FRAME if conf is None else conf['hop_length'])[0]
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
        # return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc)))
        return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))

    except Exception as error:
        print('bad file', error)
        return pd.Series([0] * 210)
        # return pd.Series([0] * 198)


def get_mfcc(name, path):
    """ Used for train model """
    data = get_mfcc_data(name, path)

    return get_mfcc_feature(data)


# Features from LightGBM baseline kernel: https://www.kaggle.com/opanichev/lightgbm-baseline
# MAPk from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def extract_feature(data):
    features = {}

    abs_data = np.abs(data)
    diff_data = np.diff(data)

    n = 1
    features = calc_part_features(data, features, n=n)
    features = calc_part_features(abs_data, features, n=n, prefix='abs_')
    features = calc_part_features(diff_data, features, n=n, prefix='diff_')

    n = 2
    features = calc_part_features(data, features, n=n)
    features = calc_part_features(abs_data, features, n=n, prefix='abs_')
    features = calc_part_features(diff_data, features, n=n, prefix='diff_')

    n = 3
    features = calc_part_features(data, features, n=n)
    features = calc_part_features(abs_data, features, n=n, prefix='abs_')
    features = calc_part_features(diff_data, features, n=n, prefix='diff_')

    return features


def calc_part_features(data, features, n=2, prefix='', f_i=1):
    for i in range(0, len(data), len(data) // n):
        features['{}mean_{}_{}'.format(prefix, f_i, n)] = np.mean(data[i:i + len(data) // n])
        features['{}std_{}_{}'.format(prefix, f_i, n)] = np.std(data[i:i + len(data) // n])
        features['{}min_{}_{}'.format(prefix, f_i, n)] = np.min(data[i:i + len(data) // n])
        features['{}max_{}_{}'.format(prefix, f_i, n)] = np.max(data[i:i + len(data) // n])

    return features


def extract_features(files, path, fname='fname'):
    features = {}

    for f in tqdm(files):
        features[f] = {}

        fs, data = scipy.io.wavfile.read(os.path.join(path, f))

        features[f]['len'] = len(data)
        if len(data) > 0:
            features[f] = extract_feature(data)

    features = pd.DataFrame(features).T.reset_index()
    features.rename(columns={'index': fname}, inplace=True)

    return features
