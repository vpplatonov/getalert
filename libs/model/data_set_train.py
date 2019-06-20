# https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
import numpy as np
import pandas as pd
import asyncio
import botocore
import aiobotocore

import os
import io
import pickle

import argparse
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from predict.feature_engineer import NUM_PCA, MODEL_TYPE, read_audio, \
    get_mfcc_feature, conf_load, FOLDER, audio_load
from xgboost import XGBClassifier
from pathlib import Path
from .xgboost_train import balance_class_by_over_sampling, print_class_balance, xgboost_grid_search, \
    MAX_DEPTH, MIN_CHILD_WEIGHT, GAMMA, SUBSAMPLE, COLSAMPLE_BYTREE, LEARNING_RATE

import boto3
from model.xgboost_db_save import COLLECTION_FILE, CLASS_PREDICTED, DB_NAME, MIN_IO, aws_secret_access_key, aws_access_key_id
from model.feed_model_store import get_db, FEED_TEST

tqdm.pandas()

PATH_SUFFIX_LOAD = '../'
# PATH_SUFFIX_LOAD = '../ESC-50-master/'
# PATH_SUFFIX_SAVE = '../ESC-50-master/'
PATH_SUFFIX_SAVE = '../'


def data_set_load(test_size=0.2, random_state=42, isPCA=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='{}/../{}output/dataset/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_LOAD
                        ))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    print(load_path)

    # Load features
    # Load from numpy binary format
    X = np.load(os.path.join(load_path, 'dataset.npy'))
    y = np.load(os.path.join(load_path, 'labels.npy'))

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        shuffle=True)

    print(X_train.shape)
    print(X_test.shape)

    if isPCA:
        np.save(os.path.join(load_path, 'train_dataset.npy'), X_train, fix_imports=False)

    return X_train, X_test, y_train, y_test


# Data utilities
def datapath(conf, filename):
    return Path(conf['folder']) / filename


def loaddata(path, filename):
    return np.load(path / filename, allow_pickle=True)


def samplewise_mean_X(X):
    for i in range(len(X)):
        X[i] -= np.mean(X[i], keepdims=True)
        X[i] /= (np.std(X[i], keepdims=True) + 1.0)  # Kind of Compressor effect


def data_set_load_folder(dataroot, prefix=''):
    return loaddata(dataroot, '{}X_train.npy'.format(prefix)), \
           loaddata(dataroot, '{}y_train.npy'.format(prefix)), \
           loaddata(dataroot, '{}idx_train.npy'.format(prefix)), \
           loaddata(dataroot, '{}plain_y_train.npy'.format(prefix))


def data_set_limit_apply(X, y, limit, lcolumn):
    """
    For well known sound possible using limited data set
    :param X:
    :param y:
    :param limit:
    :param lcolumn:
    :return:
    """
    df = pd.DataFrame(data=X)
    df[lcolumn] = y
    # print(df.head())

    if limit > 0:
        df = df.groupby(lcolumn).apply(lambda x: x[:limit])
    counter = df[lcolumn].value_counts(normalize=False)
    # print(counter)
    y = df[lcolumn].values
    X = df.drop([lcolumn], axis=1).values

    return X, y, counter


def get_class_id(sample, conf):

    # class_id = {11: 'custom_fid'}
    # class_id = {5: 'human_non_speech'}
    class_ids = [{2: 'domestic'} for a in range(len(sample))]

    return class_ids


def get_s3_files():
    """
    READ FROM Mongo DB file info
    """
    collection = get_db(db_name=DB_NAME)[COLLECTION_FILE]
    query = {'feed_id': FEED_TEST,
             'status': {'$in': [0, 1]},
             'class_predicted': CLASS_PREDICTED}
    selected_field = {'filename': 1, '_id': 0}
    fp_sounds = collection.find(query, selected_field)
    # print(fp_sounds)
    # convert to list()
    return fp_sounds


async def get_s3_file(loop, filename):
    """
    Read from S3 bucket MIN IO filename from Mongo DB
    :param s3_client:
    :param filename:
    :return:
    """
    file = filename['filename'].split('/')

    session = aiobotocore.get_session(loop=loop)
    async with session.create_client(
        # The name of the service for which a client will be created.
        service_name="s3",
        endpoint_url=MIN_IO,
        aws_secret_access_key=aws_secret_access_key,
        aws_access_key_id=aws_access_key_id
    ) as s3_client:

        try:
            s3_data = await s3_client.get_object(
                Bucket=file[0],
                Key=file[1] + '/' + file[2]
            )
            body = await s3_data["Body"].read()
            data = io.BytesIO(body)
            return data
        except:
            return b''


def data_set_load_cnn_data(load_path, test_size=0.2, random_state=42, limit=0, s3_load=True):

    DATAROOT = Path('./../../GetAlertCNN/GetAlertCNN')
    DATAROOT_EXTRA = ['donateacry-corpus', 'getalert', 'AudioTagging',
                      'FreesoundScream', 'pond5', 'UrbanSound8K',
                      'freesound-audio-tagging-2019']

    FNAME_COLUMN = 'filename'
    LNAME_COLUMN = 'category'

    conf = conf_load(DATAROOT, folder=FOLDER)

    if os.path.exists(os.path.join('../output/', FOLDER, 'X_train.npy')):
        X_train, y_train, idx_train, plain_y_train = data_set_load_folder(Path('../output') / FOLDER,
                                                                          prefix='')
        print("will be used already prepared data set for", FOLDER)
        X_test = None
        i2c = np.load(os.path.join(load_path, FOLDER, 'to_labels.npy')).tolist()
        print(X_train.shape)
        print(y_train.shape)

        # Next step
        # Add exclude file to train with class
        if s3_load:
            files = get_s3_files()
            print("Add files from MongoDB")
        else:
            # folder = os.path.normpath("c:/Users/User/Downloads/Skype/_false_positive")
            folder = os.path.normpath("../cnn_predicted_1_cry")
            files = [i for i in os.listdir(folder)]
            print('Adding from:', folder)

        loop = asyncio.get_event_loop()
        for i, file_to_filter in enumerate(files):
            if s3_load:
                file = loop.run_until_complete(get_s3_file(loop, file_to_filter))
                data = audio_load(conf, pathname=file, pydub_read=True)
            else:
                x = read_audio(conf, pathname=os.path.join(folder, file_to_filter))
                xx = get_mfcc_feature(x)
                data = [xx]

            class_ids = get_class_id(data, conf)
            class_ids_keys = []
            for class_id in class_ids:
                class_id_keys = list(class_id.keys())[0]
                class_ids_keys.append(class_id_keys)
                if class_id_keys not in i2c.keys():
                    print('New class', class_id)
                    i2c.update(class_id)
                    np.save(os.path.join(load_path, FOLDER, 'to_labels.npy'), i2c, fix_imports=False)

            X_train = np.concatenate((X_train, np.array(data)), axis=0)
            y_train = np.concatenate((y_train, np.array(class_ids_keys)), axis=0)

        print('after adding filter sample', X_train.shape, y_train.shape)

    else:
        for conf in [conf]:
            print('== Attempt [%s] ==' % conf['folder'])

            # a. Load all dataset -> all_(X4|y|idx)_train, (X4|idx)_test
            all_X_train, all_y_train, all_idx_train, plain_y_train = data_set_load_folder(DATAROOT / conf['folder'])

            print(np.unique(all_y_train))
            # Concatenate
            print(all_X_train.shape)

            X_test, idx_test = loaddata(DATAROOT / (conf['folder']), 'X_test.npy'), \
                               loaddata(DATAROOT / (conf['folder']), 'idx_test.npy')

            label2int = loaddata(DATAROOT / (conf['folder']), 'i2label.npy').tolist()
            labels = loaddata(DATAROOT / (conf['folder']), 'labels.npy')

            for dataroot_extra in DATAROOT_EXTRA:
                print('== add extra [%s] ==' % dataroot_extra)
                dataroot_extra_path = Path('.') / '..\..' / dataroot_extra
                # print(os.path.abspath(str(dataroot_extra_path / conf['folder'] / 'X_train.npy')))

                if os.path.exists(os.path.abspath(str(dataroot_extra_path / conf['folder'] / 'X_train.npy'))):
                    all_X_train_extra, all_y_train_extra,\
                    all_idx_train_extra, plain_y_train_extra = data_set_load_folder(dataroot_extra_path / conf['folder'])
                    print(np.unique(all_y_train_extra))

                    # Concatenate
                    print(all_X_train_extra.shape)
                    all_X_train = np.concatenate((all_X_train, all_X_train_extra), axis=0)
                    all_y_train = np.concatenate((all_y_train, all_y_train_extra), axis=0)
                    max = np.amax(all_idx_train)
                    all_idx_train_extra = np.array([i + max + 1 for i in all_idx_train_extra])
                    all_idx_train = np.concatenate((all_idx_train, all_idx_train_extra), axis=0)
                    plain_y_train = np.concatenate((plain_y_train, plain_y_train_extra), axis=0)

                if os.path.exists(os.path.abspath(str(dataroot_extra_path / conf['folder'] / 'X_test.npy'))):
                    X_test_extra, idx_test_extra = np.load(dataroot_extra_path / conf['folder'] / 'X_test.npy',
                                                           allow_pickle=True), \
                                                   np.load(dataroot_extra_path / conf['folder'] / 'idx_test.npy',
                                                           allow_pickle=True)

                    X_test = np.concatenate((X_test, X_test_extra), axis=0)
                    max = np.amax(idx_test)
                    idx_test_extra = np.array([i + max + 1 for i in idx_test_extra])
                    idx_test = np.concatenate((idx_test, idx_test_extra), axis=0)

                if os.path.exists(os.path.abspath(str(dataroot_extra_path / conf['folder'] / 'labels.npy'))):
                    label2int_extra = np.load(dataroot_extra_path / conf['folder'] / 'i2label.npy',
                                              allow_pickle=True).tolist()
                    print(label2int_extra)
                    label2int.update(label2int_extra)
                    labels_extra = np.load(dataroot_extra_path / conf['folder'] / 'labels.npy', allow_pickle=True)
                    labels = np.concatenate((labels, labels_extra), axis=0)

            num_classes = len(labels)

            # Next line for 2D conv only
            # all_y_train = to_categorical(all_y_train)

            # a. Normalize samplewise if requested
            if conf['normalize'] == 'samplewise':
                print(' normalize samplewise')
                samplewise_mean_X(all_X_train)
                samplewise_mean_X(X_test)

            all_X_train = np.squeeze(all_X_train, axis=2)
            X_test = np.squeeze(X_test, axis=2)

            # limit
            all_X_train, all_y_train, counter = data_set_limit_apply(all_X_train, all_y_train, limit, LNAME_COLUMN)
            print(counter)

            X_train, y_train, idx_train = all_X_train, all_y_train, all_idx_train
            print('Filtered samples on blacklist, now trainset has %d samples' % len(idx_train))

            i2c = {}
            for c in label2int.keys():
                i2c[label2int[c]] = c

            # create folder for save dataset
            # save_path = os.path.join(load_path, FOLDER)
            # shutil.rmtree(str(save_path))
            # os.makedirs(str(save_path))

            np.save(os.path.join(load_path, FOLDER, 'to_labels.npy'), i2c, fix_imports=False)
            np.save(os.path.join(load_path, FOLDER, 'X_train.npy'), X_train, fix_imports=False)
            np.save(os.path.join(load_path, FOLDER, 'y_train.npy'), y_train, fix_imports=False)
            np.save(os.path.join(load_path, FOLDER, 'idx_train.npy'), idx_train, fix_imports=False)
            np.save(os.path.join(load_path, FOLDER, 'plain_y_train.npy'), plain_y_train, fix_imports=False)
            np.save(os.path.join(load_path, FOLDER, 'conf.npy'), conf, fix_imports=False)

    return X_train, X_test, y_train, None, i2c  # y_test


def model_train(X_train, X_val, y_train, y_val, i2c, save_path, model_type='SVC', class_balance=False, grid_search=True):

    print("Model type: ", model_type)

    if model_type == 'SVC':
        clf = SVC(kernel='rbf', probability=True, gamma='scale')

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        print(accuracy_score(y_pred, y_val))
        # print(f1_score(y_pred, y_val))

        # clf = SVC(kernel='rbf', probability=True, C=4, gamma=0.01)
        # clf = SVC(kernel='rbf', probability=True, gamma='scale')
        #
        # clf.fit(X_train, y_train)
        #
        # print(accuracy_score(clf.predict(X_test), y_test))

        # Define the paramter grid for C from 0.001 to 10, gamma from 0.001 to 10
        C_grid = [4, 6, 8, 10]
        gamma_grid = [0.005]  # [0.001, 0.005, 0.01]
        param_grid = {'C': C_grid, 'gamma': gamma_grid}

        grid = GridSearchCV(SVC(kernel='rbf', probability=True, gamma='auto'),
                            param_grid,
                            cv=10,
                            scoring="accuracy")
        grid.fit(X_train, y_train)

        performance = grid.best_score_
        parameters = grid.best_params_

        # Find the best model
        print(performance)
        print(parameters)
        print(grid.best_estimator_)

        # Save performances
        with open(os.path.join(save_path, 'performance.json'), 'w') as fp:
            json.dump(performance, fp)

        # Save parameters
        with open(os.path.join(save_path, 'parameters.json'), 'w') as fp:
            json.dump(parameters, fp)

        # final model
        clf = SVC(kernel='rbf', C=parameters['C'], gamma=parameters['gamma'], probability=True)

    else:

        labels = [c[1] for c in i2c.items()]
        # print('labels', labels)
        print_class_balance('Current fold category distribution', y_train, labels)
        _X_train, X_val, _y_train, y_val = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            shuffle=True)

        if class_balance:
            print_class_balance('Current fold category distribution', _y_train, labels)
            _X_train, _y_train = balance_class_by_over_sampling(_X_train, _y_train)
            print_class_balance('after balanced', _y_train, labels)

        # https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning
        # max_depth = 5 : This should be between 3-10. I’ve started with 5 but you can choose
        #   a different number as well. 4-6 can be good starting points.
        # min_child_weight = 1 : A smaller value is chosen because it is a highly imbalanced class problem
        #   and leaf nodes can have smaller size groups.
        # gamma = 0 : A smaller value like 0.1-0.2 can also be chosen for starting. This will anyways be tuned later.
        # subsample, colsample_bytree = 0.8 : This is a commonly used used start value.
        #   Typical values range between 0.5-0.9.
        # scale_pos_weight = 1: Because of high class imbalance.

        # XGBoost from https://www.kaggle.com/amlanpraharaj/xgb-using-mfcc-opanichev-s-features-lb-0-811
        # clf = XGBClassifier(max_depth=5,
        #                     learning_rate=0.05,
        #                     n_estimators=3000)

        base_params = {
            MAX_DEPTH: 7,
            MIN_CHILD_WEIGHT: 5,
            GAMMA: 0.4,
            SUBSAMPLE: 0.7,
            COLSAMPLE_BYTREE: 0.75,
            LEARNING_RATE: 0.015
        }

        if os.path.exists(os.path.join(save_path, 'parameters.json')):
            with open(os.path.join(save_path, 'parameters.json'), 'r') as fp:
                base_params = json.load(fp)

            grid_search = False

        print('base_params', base_params)

        clf = XGBClassifier(learning_rate=base_params[LEARNING_RATE],
                            n_estimators=1000,
                            max_depth=base_params[MAX_DEPTH],
                            min_child_weight=base_params[MIN_CHILD_WEIGHT],
                            gamma=base_params[GAMMA],
                            subsample=base_params[SUBSAMPLE],
                            colsample_bytree=base_params[COLSAMPLE_BYTREE],
                            colsample_bylevel=0.9,
                            reg_alpha=0.2,
                            nthread=4,
                            scale_pos_weight=1,
                            seed=27,
                            objective='multi:softmax',
                            num_class=len(labels))

        # print(_X_train.shape)
        # print(_y_train.shape)

        # clf.fit(_X_train, _y_train, verbose=False)

        if grid_search:
            clf.fit(_X_train, _y_train,
                    verbose=False,
                    early_stopping_rounds=2,
                    eval_set=[(X_val, y_val)])
        else:
            clf.fit(X_train, y_train, verbose=False)

        # Performance sur le train
        # print('train', accuracy_score(clf.predict(_X_train), _y_train))
        # print(X_val.shape)

        if y_val is not None and len(y_val) == X_val.shape[0]:
            # print('test X', X_val.shape)
            # print('test y', len(y_val))
            print(accuracy_score(clf.predict(X_val), y_val))

        if grid_search:
            params_trained = xgboost_grid_search(X_train, y_train, base_params)

            # Save parameters
            with open(os.path.join(save_path, 'parameters.json'), 'w') as fp:
                json.dump(params_trained, fp)

    return clf


def main():
    local_dataset = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',
                        default='{}/../{}output/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_SAVE
                        ))
    parser.add_argument('--load_path_label',
                        default='{}/../{}output/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_LOAD
                        ))

    parser.add_argument('--load_path',
                        default='{}/../{}output/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_LOAD
                        ))

    # Arguments
    args = parser.parse_args()
    save_path = os.path.normpath(args.save_path)
    load_path_label = os.path.normpath(args.load_path_label)
    load_path = os.path.normpath(args.load_path)

    i2c = np.load(os.path.join(load_path_label, 'dataset', 'to_labels.npy')).tolist()

    if local_dataset:
        X_train, X_test, y_train, y_test = data_set_load(test_size=0.1,
                                                         random_state=42)

        print(X_train.shape)
        print(y_train.shape)
    else:
        X_train, X_test, y_train, y_test, i2c = data_set_load_cnn_data(load_path=load_path,
                                                                       test_size=0.1,
                                                                       random_state=42,
                                                                       limit=800)
        # print(i2c)
        X_pca = X_train
        y_pca = y_train
        exit(0)

    X_val = X_test
    y_val = y_test
    X_test_pca = X_test
    model_type = MODEL_TYPE

    # # Apply scaling for PCA
    if model_type == 'SVC':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply PCA for dimension reduction
        pca = PCA(n_components=NUM_PCA).fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        print(sum(pca.explained_variance_ratio_))

        y_pca = y_train

        # Fit an SVM model
        X_train, X_val, y_train, y_val = train_test_split(X_pca, y_train,
                                                          test_size=0.1,
                                                          random_state=42,
                                                          shuffle=True)

    # model train model_type='SVC' / 'XGBoost'
    clf = model_train(X_train, X_val, y_train, y_val,
                      i2c,
                      os.path.join(save_path, 'model' if model_type == 'SVC' else FOLDER),
                      model_type=model_type)

    if model_type != 'SVC' and local_dataset:
        X_pca = np.load(os.path.join(load_path, 'dataset.npy'))
        y_pca = np.load(os.path.join(load_path, 'labels.npy'))

    # final model training on entire data
    # print('final model training ...')
    # clf.fit(X_pca, y_pca)
    # print('train', accuracy_score(clf.predict(X_pca), y_pca))

    if y_test is not None:
        y_pred = clf.predict(X_test_pca)
        print('test', accuracy_score(y_pred, y_test))

    # Save model
    with open(os.path.join(save_path, 'model' if model_type == 'SVC' else FOLDER, 'model.pkl'), 'wb') as fp:
        pickle.dump(clf, fp)

    # str_preds, _ = convert_to_labels(clf.predict_proba(X_test_pca), i2c, k=1)
    #
    # # Write to outputs
    # subm = pd.DataFrame()
    # # subm[FNAME_COLUMN'] = y_test
    # subm[LNAME_COLUMN] = str_preds
    # subm.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
