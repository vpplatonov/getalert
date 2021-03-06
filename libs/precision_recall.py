# -*- coding: utf-8 -*-

import sys
import argparse
import logging
import os
import re
import timeit
import numpy as np
import pandas as pd

import shutil

from predict.audio_predict import get_file_name, model_init, audio_load, play_list_predict
from predict.strategy import predict_category

DESTINATION = 'predicted'
isPCA = False


def main():

    save_path, load_path_data, load_path_model, load_path_label, file_name = get_file_name()
    model, scaler, pca, i2c = model_init(load_path_model, load_path_label)

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        # default='{}/../../baby_cry_detection/data'.format(os.path.dirname(os.path.abspath(__file__)))
                        default='{}/../../freesound-audio_origin-tagging-2019'.format(os.path.dirname(os.path.abspath(__file__)))
                        )

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)

    # READ FILES IN SUB-FOLDERS of load_path and FEATURE ENGINEERING

    # list load_path sub-folders
    regex = re.compile(r'^train_curated$')
    read_from_csv = True
    # regex = re.compile(r'^donateacry-ios.+')
    directory_list = [i for i in os.listdir(load_path) if regex.search(i)]
    # directory_list = [i for i in os.listdir(load_path)]

    # initialise empty array for labels
    # y = []

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), DESTINATION)
    # print(dest)
    category_checking = 'Marimba_and_xylophone'

    # iteration on sub-folders

    for directory in directory_list:
        # initialise empty array for labels
        y = []
        i = 0

        if read_from_csv:
            # For big amount of files use csv info for reduce file checking
            print(directory)
            df = pd.read_csv(os.path.join(load_path, directory + '.csv'))
            # should be corrected for different dataset
            # category_checking = 'Screaming'
            # category_checking = ['Child_speech_and_kid_speaking',
            #                      'Female_singing',
            #                      'Female_speech_and_woman_speaking',
            #                      'Male_singing',
            #                      'Male_speech_and_man_speaking']
            print('Category checking: ', category_checking)
            file_list = [fname for fname in df[df['labels'].isin([category_checking])]['fname']]
            print('files:', len(file_list))
        else:
            file_list = os.listdir(os.path.join(load_path, directory))

        # iteration on people_noise-ios-mix files in each sub-folder
        for audio_file in file_list:
            # file_reader = Reader(os.path.join(load_path, directory, audio_file))
            # iOS:
            # 0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu.caf
            # app instance uuid (36 chars)-unix epoch timestamp-app version-gender-age-reason
            # Android:
            # 0c8f14a9-6999-485b-97a2-913c1cbf099c-1431028888092-1.7-m-26-sc.3gp
            # The structure is the same with the exception that the unix epoch timestamp is in milliseconds

            play_list_processed = audio_load(os.path.join(load_path, directory), audio_file)
            if isPCA:
                play_list_processed = scaler.transform(play_list_processed)
                play_list_processed = pca.transform(play_list_processed)
            predictions = play_list_predict(model, i2c, play_list_processed, k=1)

            # Voting strategy - must be changed to first success
            #     Full - all category the same in first place
            #     Half - as min as half in first place
            #     Panic - even if selected category present in second place
            pred = predict_category(predictions,
                                    category=category_checking,
                                    strategy='Once',
                                    threshold=0.15)

            # X4full = np.concatenate((X4full, avg_features), axis=0)
            y.append((audio_file, pred))
            print((audio_file, predictions))
            # if pred == '1':
            #     print((audio_file, pred))
            #     shutil.copy(os.path.join(load_path, directory, audio_file), dest)
            i += 1
            if i > 100:
                break

        y_recall = [file for file, i in y if i == '1']
        recall = len(y_recall) / len(y)

        print(directory)
        print(recall)


if __name__ == '__main__':
    main()
