import os
import math
import librosa
import logging

import numpy as np
import pandas as pd

from scipy.stats import skew


logger = logging.getLogger('app.py')


async def play_list_predict(model, i2c, play_list_processed, conf):

    # skip_sound = False

    for signal in play_list_processed:
        predict = model.predict_proba([signal])
        logger.debug(('predict --->>>', predict))
        # get id with max score
        max_id = predict[0].argsort()[-1]
        # get the max pred score
        pred_score = predict[0][max_id]
        # get the predicted class name
        pred_class = i2c.get(max_id)
        # check if class and threshold is passing
        logger.debug({'max_id': max_id, 'pred_score': pred_score, 'pred_class': pred_class})
        if pred_class == conf['fp_class'] and pred_score >= conf['fp_threshold']:
            return True

    return False


async def get_samples(samples, input_length, conf={}):

    play_list = list()
    
    logger.debug({'min': samples.min(), 'mean': samples.mean(), 'max': samples.max()})
    # normalize the samples
    data = np.array([(s / 2 ** 16.0) * 2 for s in samples])
    # data = samples[:].astype(np.float32)

    logger.debug({'min': samples.min(), 'mean': samples.mean(), 'max': samples.max()})
    logger.debug({'min': data.min(), 'mean': data.mean(), 'max': data.max()})

    logger.debug([len(data), input_length])

    # generate chunks and get the features
    dur = len(data)
    # window cover
    cover = input_length // 2
    
    if input_length > dur:
        max_offset = input_length - dur
        middle = int(max_offset * 0.5)
        # center the sample and pad with zeros
        temp_sample = np.pad(data, (middle, max_offset - middle), "constant")
        # get the features
        temp_featue = await get_mfcc_feature(temp_sample)
        # save the features
        play_list.append(temp_featue)
    else:
        for i in range(0, dur, cover):
            if i + input_length <= dur:
                # get the sample
                temp_sample = data[i:i + input_length]
                # get the features
                temp_featue = await get_mfcc_feature(temp_sample)
                # save the features
                play_list.append(temp_featue)
            elif i + input_length > dur:
                # get the sample
                temp_sample = data[dur - input_length:dur]
                # get the features
                temp_featue = await get_mfcc_feature(temp_sample)
                # save the features
                play_list.append(temp_featue)
                break

    return play_list


async def get_mfcc_feature(data, conf={}):
    """ Generate mfcc features with mean and standard deviation
        all librosa features have hop_length=512 by default
    """
    try:
        ft1 = librosa.feature.mfcc(data,
                                   sr=conf.get('sampling_rate', 16000),
                                   n_mfcc=conf.get('n_mfcc', 30))
        ft2 = librosa.feature.zero_crossing_rate(data,
                                                 hop_length=conf.get('hop_length', 160))[0]
        ft3 = librosa.feature.spectral_rolloff(data,
                                               sr=conf.get('sampling_rate', 16000),
                                               hop_length=conf.get('hop_length', 160))[0]
        ft4 = librosa.feature.spectral_centroid(data,
                                                sr=conf.get('sampling_rate', 16000),
                                                hop_length=conf.get('hop_length', 160))[0]
        ft5 = librosa.feature.spectral_contrast(data,
                                                sr=conf.get('sampling_rate', 16000),
                                                n_bands=6,
                                                fmin=200.0)[0]
        ft6 = librosa.feature.spectral_bandwidth(data,
                                                 sr=conf.get('sampling_rate', 16000),
                                                 hop_length=conf.get('hop_length', 160))[0]
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

    except Exception as error:
        logging.exception("Exception occurred")
        return pd.Series([0] * 210)
