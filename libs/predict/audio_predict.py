# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import warnings
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import librosa

from predict.feature_engineer import (
    SAMPLE_RATE, get_mfcc_feature, convert_to_labels, NUM_PCA,
    PATH_SUFFIX_LOAD, PATH_SUFFIX_SAVE
)
from predict.strategy import predict_category

SOUND_DURATION = 5.0
SOUND_RANGE = 1
# For Docker env
PATH_SUFFIX = '/opt/ml/'


def get_file_name():
    parser = argparse.ArgumentParser()
    base_path = os.path.dirname(os.path.abspath(__file__))
    # base_path = PATH_SUFFIX

    parser.add_argument('--load_path_data',
                        default='{}/../audio_samples/'.format(base_path))
    parser.add_argument('--load_path_model',
                        default='{}output/model/'.format(
                            PATH_SUFFIX
                        ))
    parser.add_argument('--load_path_label',
                        default='{}output/dataset/'.format(
                            PATH_SUFFIX
                        ))
    parser.add_argument('--file_name', default='V_2017-04-01+08_04_36=0_13.mp3')

    parser.add_argument('--save_path',
                        default='{}output/prediction/'.format(
                            PATH_SUFFIX
                        ))

    # Arguments
    args = parser.parse_args()

    return os.path.normpath(args.save_path), os.path.normpath(args.load_path_data), os.path.normpath(args.load_path_model),\
        os.path.normpath(args.load_path_label), args.file_name


def model_init(load_path_model, load_path_label):
    # https://stackoverflow.com/questions/41146759/check-sklearn-version-before-loading-model-using-joblib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
            model = pickle.load(fp)

    i2c = np.load(os.path.join(load_path_label, 'to_labels.npy')).tolist()
    X_train = np.load(os.path.join(load_path_label, 'train_dataset.npy'))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    # Apply PCA for dimension reduction
    pca = PCA(n_components=NUM_PCA).fit(X_scaled)

    return model, scaler, pca, i2c


def audio_load(load_path_data, file_name):
    play_list = list()

    logging.debug('audio_load')

    # load audio with different length
    # https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data from Data Generator
    # Read and Resample the audio
    data, _ = librosa.core.load(os.path.join(load_path_data, file_name),
                                sr=SAMPLE_RATE,
                                # res_type='kaiser_fast'
                                )

    input_length = SAMPLE_RATE * SOUND_DURATION

    # Random offset / Padding
    if len(data) > input_length:
        # max_offset = len(data) - input_length
        # offset = np.random.randint(max_offset)
        # end = int(input_length + offset)
        # data = data[offset:end]

        for offset in range(math.floor((len(data) - input_length) / SAMPLE_RATE) + 1):
            # Feature extraction
            tmp = get_mfcc_feature(data[(offset * SAMPLE_RATE):int(offset * SAMPLE_RATE + input_length)])
            play_list.append(tmp)
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, int(input_length - len(data) - offset)), "constant")
        # Feature extraction
        tmp = get_mfcc_feature(data)
        play_list.append(tmp)

    # # audio_length = sampling_rate * audio_duration
    # for offset in range(SOUND_RANGE):
    #     audio_data, _ = librosa.load(os.path.join(load_path_data, file_name),
    #                                  sr=SAMPLE_RATE,
    #                                  mono=True,
    #                                  offset=offset,
    #                                  duration=SOUND_DURATION)
    #     # Feature extraction
    #     tmp = get_mfcc_feature(audio_data)
    #     play_list.append(tmp)

    return play_list


def play_list_predict(model, i2c, play_list_processed):
    predictions = list()

    for signal in play_list_processed:
        predict = model.predict_proba([signal])
        str_preds, idx = convert_to_labels(predict, i2c)
        predictions.append(dict(zip(str_preds[0].split(' '), predict[0][idx[0]])))

    return predictions


def audio_prediction():

    save_path, load_path_data, load_path_model, load_path_label, file_name = get_file_name()
    model, scaler, pca, i2c = model_init(load_path_model, load_path_label)

    play_list_processed = audio_load(load_path_data, file_name)
    play_list_processed = scaler.transform(play_list_processed)
    play_list_processed = pca.transform(play_list_processed)
    predictions = play_list_predict(model, i2c, play_list_processed)
    # print(predictions)

    # Voting strategy - must be changed to first success
    #     Full - all category the same in first place
    #     Half - as min as half in first place
    #     Panic - even if selected category present in second place
    pred = predict_category(predictions,
                            category='crying_baby',
                            strategy='Panic')

    # Save prediction result
    with open(os.path.join(save_path, 'prediction.txt'), 'w') as text_file:
        text_file.write(f"{pred}")

    return pred


if __name__ == '__main__':
    audio_prediction()
