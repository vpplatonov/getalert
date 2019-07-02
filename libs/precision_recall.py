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

from predict.audio_predict import get_file_name, model_init, audio_load_extra, play_list_predict
from predict.strategy import predict_category
from pathlib import Path
from predict.feature_engineer import conf_load, FOLDER, audio_load, MODEL_TYPE

DESTINATION = 'predicted'
isPCA = MODEL_TYPE == 'SVC'


def main():

    save_path, load_path_data, load_path_model, load_path_label, file_name = get_file_name()
    model, scaler, pca, i2c = model_init(load_path_model, load_path_label, load_model_db=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='c:/Users/User/Downloads/Skype'
                        # default='{}/../../getalert'.format(os.path.dirname(os.path.abspath(__file__)))
                        # default='{}/../../donateacry-corpus'.format(os.path.dirname(os.path.abspath(__file__)))
                        # default='{}/../../ESC-50'.format(os.path.dirname(os.path.abspath(__file__)))
                        # default='{}/../../freesound-audio-tagging-2019'.format(os.path.dirname(os.path.abspath(__file__)))
                        # default='{}/../../UrbanSound8K'.format(os.path.dirname(os.path.abspath(__file__)))
                        )

    parser.add_argument('--load_path_model',
                        default='{}/../output/'.format(
                            os.path.dirname(os.path.abspath(__file__))
                        ))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    print('load_path', load_path)

    dataroot = Path('./../output')
    conf = conf_load(dataroot, folder=FOLDER)
    print(conf)

    # READ FILES IN SUB-FOLDERS of load_path and FEATURE ENGINEERING

    # list load_path sub-folders
    # regex = re.compile(r'^train_curated$')
    regex = re.compile(r'^_false.+')
    # regex = re.compile(r'^predicted$')
    # regex = re.compile(r'^cnn_predicted_1_c.+')
    # regex = re.compile(r'^cnn_predicted_c.+')
    # regex = re.compile(r'^audio$')
    # regex = re.compile(r'^help-trained$')
    # regex = re.compile(r'^baby-cry-veri.+')
    read_from_csv = False

    # regex = re.compile(r'^donateacry-ios.+')
    directory_list = [i for i in os.listdir(load_path) if regex.search(i)]
    # directory_list = [i for i in os.listdir(load_path)]

    # initialise empty array for labels
    # y = []

    # dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), DESTINATION)
    dest = Path('./..') / DESTINATION
    # print(dest)
    shutil.rmtree(str(dest))
    os.makedirs(str(dest))
    # category_checking = 'Marimba_and_xylophone'
    # category_checking = 'crying_baby'
    # category_checking = 'custom_fid'
    category_checking = 'domestic'
    # category_checking = 'human_non_speech'
    print(category_checking)

    # iteration on sub-folders

    for directory in directory_list:
        # initialise empty array for labels
        y = []

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
        i = j = 0
        print(directory)
        print('Total files', len(file_list))
        counter = {}
        for audio_file in file_list:

            # play_list_processed = audio_load_extra(os.path.join(load_path, directory), audio_file)
            play_list_processed = audio_load(conf,
                                             os.path.join(load_path, directory, audio_file),
                                             pydub_read=True)
            if isPCA:
                play_list_processed = scaler.transform(play_list_processed)
                play_list_processed = pca.transform(play_list_processed)
            predictions = play_list_predict(model, i2c, play_list_processed, k=1)

            # Voting strategy - must be changed to first success
            pred = predict_category(predictions,
                                    category=category_checking,
                                    strategy='Once',
                                    threshold=0.35)

            # X4full = np.concatenate((X4full, avg_features), axis=0)
            # y.append((audio_file, pred))

            print((audio_file, predictions))
            if pred == '1':
                # print((audio_file, predictions))
                i += 1
                # shutil.copy(os.path.join(load_path, directory, audio_file), str(dest))
            else:
                for p in predictions:
                    if list(p.keys())[0] in list(counter.keys()):
                        counter[list(p.keys())[0]] += 1
                    else:
                        counter[list(p.keys())[0]] = 1
            j = j + 1

            # if j > 10:
            #     break

        recall = float(i / j)
        print(recall)
        print(counter)
        print({key: counter[key] for key in sorted(counter, key=counter.__getitem__, reverse=True)[2::-1]})  # [2::-1]


if __name__ == '__main__':
    main()
