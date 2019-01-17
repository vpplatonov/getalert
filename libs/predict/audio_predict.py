# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import librosa

from model import SAMPLE_RATE, get_mfcc_feature, convert_to_labels, NUM_PCA, PATH_SUFFIX_LOAD, PATH_SUFFIX_SAVE
from predict.strategy import predict_category

SOUND_DURATION = 5.0
SOUND_RANGE = 5


def audio_prediction():
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
    save_path = os.path.normpath(args.save_path)
    load_path_data = os.path.normpath(args.load_path_data)
    load_path_model = os.path.normpath(args.load_path_model)
    load_path_label = os.path.normpath(args.load_path_label)
    file_name = args.file_name

    play_list = list()

    for offset in range(SOUND_RANGE):
        audio_data, _ = librosa.load(os.path.join(load_path_data, file_name),
                                     sr=SAMPLE_RATE,
                                     mono=True,
                                     offset=offset,
                                     duration=SOUND_DURATION)
        play_list.append(audio_data)

    # Feature extraction
    play_list_processed = list()

    for signal in play_list:
        tmp = get_mfcc_feature(signal)
        play_list_processed.append(tmp)

    # https://stackoverflow.com/questions/41146759/check-sklearn-version-before-loading-model-using-joblib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
            model = pickle.load(fp)

    predictions = list()
    i2c = np.load(os.path.join(load_path_label, 'to_labels.npy')).tolist()
    X_train = np.load(os.path.join(load_path_label, 'train_dataset.npy'))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    # Apply PCA for dimension reduction
    pca = PCA(n_components=NUM_PCA).fit(X_scaled)

    play_list_processed = scaler.transform(play_list_processed)
    play_list_processed = pca.transform(play_list_processed)

    for signal in play_list_processed:
        predict = model.predict_proba([signal])
        str_preds, idx = convert_to_labels(predict, i2c)
        predictions.append(dict(zip(str_preds[0].split(' '), predict[0][idx[0]])))

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
