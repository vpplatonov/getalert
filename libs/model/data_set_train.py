# https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
import numpy as np
import pandas as pd

import os
import pickle

import argparse
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from model.data_set_load import LNAME_COLUMN
from predict.feature_engineer import convert_to_labels, NUM_PCA
from xgboost import XGBClassifier

tqdm.pandas()

PATH_SUFFIX_LOAD = '../'
# PATH_SUFFIX_LOAD = '../ESC-50-master/'
# PATH_SUFFIX_SAVE = '../ESC-50-master/'
PATH_SUFFIX_SAVE = '../'
PCA = False


def data_set_load(test_size=0.2, random_state=42):
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='{}/../{}output/dataset/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_LOAD
                        ))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)

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

    np.save(os.path.join(load_path, 'train_dataset.npy'), X_train, fix_imports=False)

    return X_train, X_test, y_train, y_test


def model_train(X_train, X_val, y_train, y_val, save_path, model_type='SVC'):

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
        C_grid = [1, 4, 6, 8, 10]
        gamma_grid = [0.001, 0.005, 0.01]
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
        # XGBoost from https://www.kaggle.com/amlanpraharaj/xgb-using-mfcc-opanichev-s-features-lb-0-811
        clf = XGBClassifier(max_depth=5,
                            learning_rate=0.05,
                            n_estimators=3000,
                            n_jobs=-1,
                            random_state=0,
                            reg_alpha=0.2,
                            colsample_bylevel=0.9,
                            colsample_bytree=0.9)

        clf.fit(X_train, y_train)
        print(accuracy_score(clf.predict(X_val), y_val))

    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',
                        default='{}/../{}output/model/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_SAVE
                        ))
    parser.add_argument('--load_path_label',
                        default='{}/../{}output/dataset/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_LOAD
                        ))

    parser.add_argument('--load_path',
                        default='{}/../{}output/dataset/'.format(
                            os.path.dirname(os.path.abspath(__file__)),
                            PATH_SUFFIX_LOAD
                        ))

    # Arguments
    args = parser.parse_args()
    save_path = os.path.normpath(args.save_path)
    load_path_label = os.path.normpath(args.load_path_label)
    load_path = os.path.normpath(args.load_path)

    i2c = np.load(os.path.join(load_path_label, 'to_labels.npy')).tolist()

    X_train, X_test, y_train, y_test = data_set_load(test_size=0.2,
                                                     random_state=10)

    X_val = X_test
    y_val = y_test
    X_test_pca = X_test

    model_type = 'XGBoost'  #'SVC'

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
                                                          test_size=0.2,
                                                          random_state=10,
                                                          shuffle=True)

    # model train model_type='SVC' / 'XGBoost'
    clf = model_train(X_train, X_val, y_train, y_val, save_path, model_type=model_type)

    if model_type != 'SVC':
        X_pca = np.load(os.path.join(load_path, 'dataset.npy'))
        y_pca = np.load(os.path.join(load_path, 'labels.npy'))

    # final model training on entire data
    clf.fit(X_pca, y_pca)

    y_pred = clf.predict(X_test_pca)

    print(accuracy_score(y_pred, y_test))

    # Save model
    with open(os.path.join(save_path, 'model.pkl'), 'wb') as fp:
        pickle.dump(clf, fp)

    str_preds, _ = convert_to_labels(clf.predict_proba(X_test_pca), i2c, k=3)

    # Write to outputs
    subm = pd.DataFrame()
    # subm[FNAME_COLUMN'] = y_test
    subm[LNAME_COLUMN] = str_preds
    subm.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
