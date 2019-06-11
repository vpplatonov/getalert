# https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
import numpy as np
import pandas as pd

import os
import pickle
import shutil

import argparse
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from model.data_set_load import LNAME_COLUMN
from predict.feature_engineer import convert_to_labels, NUM_PCA, MODEL_TYPE, read_audio, \
    get_mfcc_feature, audio_load, conf_load, FOLDER
from xgboost import XGBClassifier, XGBRegressor
from pathlib import Path
from .xgboost_train import balance_class_by_over_sampling, print_class_balance

from keras.utils import to_categorical

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report, confusion_matrix

tqdm.pandas()

PATH_SUFFIX_LOAD = '../'
# PATH_SUFFIX_LOAD = '../ESC-50-master/'
# PATH_SUFFIX_SAVE = '../ESC-50-master/'
PATH_SUFFIX_SAVE = '../'

#trained parameters const
MAX_DEPTH = 'max_depth'
MIN_CHILD_WEIGHT = 'min_child_weight'
GAMMA = 'gamma'
SUBSAMPLE = 'subsample'
COLSAMPLE_BYTREE = 'colsample_bytree'
LEARNING_RATE = 'learning_rate'


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
    class_id = {2: 'domestic'}

    return class_id


def data_set_load_cnn_data(load_path, test_size=0.2, random_state=42, limit=0):

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
        # folder = os.path.normpath("c:/Users/User/Downloads/Skype/_false_positive")
        folder = os.path.normpath("../cnn_predicted_cry")
        # file = "0e55f482-b262-4bee-aeca-76a3f63dd626_20190516133957379_30676_4.001.wav"
        # files = [os.path.join(folder, file)]
        files = [i for i in os.listdir(folder)]
        X_train_extra = []
        y_train_extra = []
        print('Adding from:', folder)

        for i, file_to_filter in enumerate(files):
            x = read_audio(conf, pathname=os.path.join(folder, file_to_filter))
            print(file_to_filter)
            data = get_mfcc_feature(x)
            class_id = get_class_id(data, conf)

            if list(class_id.keys())[0] not in i2c.keys():
                i2c.update(class_id)
                print(i2c)
                np.save(os.path.join(load_path, FOLDER, 'to_labels.npy'), i2c, fix_imports=False)

            X_train_extra.append(data)
            y_train_extra.append(list(class_id.keys())[0])

        X_train = np.concatenate((X_train, np.array(X_train_extra)), axis=0)
        y_train = np.concatenate((y_train, np.array(y_train_extra)), axis=0)

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
            MAX_DEPTH: 5,
            MIN_CHILD_WEIGHT: 1,
            GAMMA: 0.1,
            SUBSAMPLE: 0.7,
            COLSAMPLE_BYTREE: 0.9,
            LEARNING_RATE: 0.05
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


def xgboost_grid_search(X_train, y_train, base_params):
    # Step 2: Tune max_depth and min_child_weight
    params_test = [{
        MAX_DEPTH: range(3, 10, 2),
        MIN_CHILD_WEIGHT: range(1, 6, 2)
    },
        {
            GAMMA: [i / 10.0 for i in range(0, 5)]
        },
        {
            SUBSAMPLE: [i / 10.0 for i in range(6, 10)],
            COLSAMPLE_BYTREE: [i / 10.0 for i in range(6, 10)]
        },
        {
            SUBSAMPLE: [i / 100.0 for i in range(65, 80, 5)],
            COLSAMPLE_BYTREE: [i / 100.0 for i in range(85, 100, 5)]
        },
        {
            LEARNING_RATE: [i / 1000.0 for i in range(5, 20, 2)]
        }
    ]

    params_trained = {}
    step = 5

    def get_param_value(param):
        return base_params[param] if param not in trained_keys else params_trained[param]

    for param_test in params_test:

        trained_keys = params_trained.keys()
        if SUBSAMPLE in param_test.keys() and SUBSAMPLE in trained_keys:
            param_test[SUBSAMPLE] = [i / 100.0 for i in range(int(params_trained[SUBSAMPLE] * 100) - step,
                                                              int(params_trained[SUBSAMPLE] * 100) + step * 2,
                                                              step)]
            param_test[COLSAMPLE_BYTREE] = [i / 100.0 for i in range(int(params_trained[COLSAMPLE_BYTREE] * 100) - step,
                                                                     int(params_trained[COLSAMPLE_BYTREE] * 100) + step * 2,
                                                                     step)]

        gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=get_param_value(LEARNING_RATE),
                                                        n_estimators=1000,
                                                        max_depth=get_param_value(MAX_DEPTH),
                                                        min_child_weight=get_param_value(MIN_CHILD_WEIGHT),
                                                        gamma=get_param_value(GAMMA),
                                                        subsample=get_param_value(SUBSAMPLE),
                                                        colsample_bytree=get_param_value(COLSAMPLE_BYTREE),
                                                        colsample_bylevel=0.9,
                                                        reg_alpha=0.2,
                                                        nthread=4,
                                                        scale_pos_weight=1,
                                                        objective='multi:softmax',
                                                        seed=27),
                                param_grid=param_test,
                                # scoring='roc_auc',
                                scoring='accuracy',
                                n_jobs=-1,
                                iid=False,
                                cv=5)

        gsearch1.fit(X_train, y_train)
        # # print(gsearch1.grid_scores_)
        print(gsearch1.best_params_, gsearch1.best_score_)
        params_trained = {**params_trained, **gsearch1.best_params_}

    return params_trained


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
        # exit(0)

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
