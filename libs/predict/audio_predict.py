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
import pandas as pd

import librosa

from predict.feature_engineer import (
    SAMPLE_RATE, get_mfcc_feature, convert_to_labels, NUM_PCA, MODEL_TYPE,
    PATH_SUFFIX_LOAD, PATH_SUFFIX_SAVE, extract_feature

)
from predict.strategy import predict_category

SOUND_DURATION = 5.0
SOUND_RANGE = 1


def get_file_name():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_data',
                        default='{}/../audio_samples/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--load_path_model',
                        default='{}/../{}output/model/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_SAVE
                        ))
    parser.add_argument('--load_path_label',
                        default='{}/../{}output/dataset/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_LOAD
                        ))
    parser.add_argument('--file_name', default='V_2017-04-01+08_04_36=0_13.mp3')

    parser.add_argument('--save_path',
                        default='{}/../{}output/prediction/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_SAVE
                        ))

    # Arguments
    args = parser.parse_args()

    return os.path.normpath(args.save_path), os.path.normpath(args.load_path_data), os.path.normpath(args.load_path_model),\
        os.path.normpath(args.load_path_label), args.file_name


def model_init(load_path_model, load_path_label, isPCA=True):
    # https://stackoverflow.com/questions/41146759/check-sklearn-version-before-loading-model-using-joblib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
            model = pickle.load(fp)

    i2c = np.load(os.path.join(load_path_label, 'XGBoost3_to_labels.npy')).tolist()
    print(i2c)

    if isPCA:
        # print(load_path_label)
        X_train = np.load(os.path.join(load_path_label, 'train_dataset.npy'))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        # Apply PCA for dimension reduction
        pca = PCA(n_components=NUM_PCA).fit(X_scaled)
    else:
        pca = scaler = None

    return model, scaler, pca, i2c


def audio_load(load_path_data, file_name, extra_features=False):
    play_list = list()

    logging.info('audio_load', file_name)

    # load prediction with different length
    # https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data from Data Generator
    # Read and Resample the prediction
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
            chank_data = data[(offset * SAMPLE_RATE):int(offset * SAMPLE_RATE + input_length)]
            tmp = get_mfcc_feature(chank_data)
            if extra_features:
                chank_data = [int((v*2**16.0)/2) for v in chank_data]
                tmp2 = pd.Series(list(extract_feature(chank_data).values()))
                play_list.append(pd.concat([tmp, tmp2]))
            else:
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

        # convert to int for extract_feature()
        if extra_features:
            data = [int((v*2**16.0)/2) for v in data]
            tmp2 = pd.Series(list(extract_feature(data).values()))
            # print(tmp2)
            play_list.append(pd.concat([tmp, tmp2]))
        else:
            play_list.append(tmp)

    return play_list


def play_list_predict(model, i2c, play_list_processed, k=2):
    predictions = list()

    for signal in play_list_processed:
        predict = model.predict_proba([signal])
        str_preds, idx = convert_to_labels(predict, i2c, k=k)
        predictions.append(dict(zip(str_preds[0].split(' '), predict[0][idx[0]])))

    return predictions


def audio_prediction():
    is_pca = MODEL_TYPE == 'SVC'
    category = 'crying_baby'
    load_model = 0  # from file / 1 from MongoDB

    print('SAMPLE_RATE', SAMPLE_RATE)
    print('SOUND_DURATION', SOUND_DURATION)
    print('category', category)
    print('model type', MODEL_TYPE)
    print('load model from', 'file' if load_model == 0 else 'MongoDB')
    save_path, load_path_data, load_path_model, load_path_label, file_name = get_file_name()

    model, scaler, pca, i2c = model_init(load_path_model, load_path_label, isPCA=is_pca)

    play_list_processed = audio_load(load_path_data, file_name, extra_features=False)
    if is_pca:
        play_list_processed = scaler.transform(play_list_processed)
        play_list_processed = pca.transform(play_list_processed)

    predictions = play_list_predict(model, i2c, play_list_processed, k=1)
    print(predictions)

    # Voting strategy - must be changed to first success
    #     Full - all category the same in first place
    #     Half - as min as half in first place
    #     Panic - even if selected category present in second place
    pred = predict_category(predictions,
                            category=category,
                            strategy='Once')

    # Save prediction result
    with open(os.path.join(save_path, 'prediction.txt'), 'w') as text_file:
        text_file.write('{}'.format(pred))

    return pred


if __name__ == '__main__':
    audio_prediction()
