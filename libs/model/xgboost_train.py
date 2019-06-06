import numpy as np
from imblearn.over_sampling import RandomOverSampler
# from .data_set_train import data_set_load
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pathlib import Path


def get_class_distribution(y):
    # y_cls can be one of [OH label, index of class, class label name]
    # convert OH to index of class
    y_cls = [np.argmax(one) for one in y] if len(np.array(y).shape) == 2 else y
    # y_cls can be one of [index of class, class label name]
    classset = sorted(list(set(y_cls)))
    # print('classset', classset)
    sample_distribution = {cur_cls:len([one for one in y_cls if one == cur_cls]) for cur_cls in classset}

    return sample_distribution


def balance_class_by_over_sampling(X, y):
    Xidx = [[xidx] for xidx in range(len(X))]
    y_cls = [np.argmax(one) for one in y] if len(np.array(y).shape) == 2 else y
    classset = sorted(list(set(y_cls)))
    sample_distribution = [len([one for one in y_cls if one == cur_cls]) for cur_cls in classset]
    nsamples = np.max(sample_distribution)
    flat_ratio = {cls:nsamples for cls in classset}

    Xidx_resampled, y_cls_resampled = RandomOverSampler(ratio=flat_ratio,
                                                        random_state=42).fit_sample(Xidx, y_cls)

    sampled_index = [idx[0] for idx in Xidx_resampled]

    return np.array([X[idx] for idx in sampled_index]), np.array([y[idx] for idx in sampled_index])


def print_class_balance(title, y, labels):

    distributions = get_class_distribution(y)
    dist_dic = {labels[cls]: distributions[cls] for cls in distributions}
    print(title, '=', dist_dic)

    zeroclasses = [label for i, label in enumerate(labels) if i not in distributions.keys()]
    if 0 < len(zeroclasses):
        print(' 0 sample classes:', zeroclasses)


def model_train(X_train, X_val, y_train, y_val, save_path):
    # https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning
    # max_depth = 5 : This should be between 3-10. I’ve started with 5 but you can choose
    #   a different number as well. 4-6 can be good starting points.
    # min_child_weight = 1 : A smaller value is chosen because it is a highly imbalanced class problem
    #   and leaf nodes can have smaller size groups.
    # gamma = 0 : A smaller value like 0.1-0.2 can also be chosen for starting. This will anyways be tuned later.
    # subsample, colsample_bytree = 0.8 : This is a commonly used used start value.
    #   Typical values range between 0.5-0.9.
    # scale_pos_weight = 1: Because of high class imbalance.

    clf = XGBClassifier(learning_rate=0.015,
                        n_estimators=1000,
                        max_depth=9,  # 5,
                        min_child_weight=1,  # 1,
                        gamma=0,
                        subsample=0.7,
                        colsample_bytree=0.9,
                        colsample_bylevel=0.9,
                        reg_alpha=0.2,
                        nthread=4,
                        scale_pos_weight=1,
                        seed=27,
                        n_jobs=-1)

    # XGBoost from https://www.kaggle.com/amlanpraharaj/xgb-using-mfcc-opanichev-s-features-lb-0-811
    # clf = XGBClassifier(max_depth=5,
    #                     learning_rate=0.05,
    #                     n_estimators=3000,
    #                     n_jobs=-1,
    #                     random_state=0,
    #                     reg_alpha=0.2,
    #                     colsample_bylevel=0.9,
    #                     colsample_bytree=0.9)

    print(X_train.shape)
    print(y_train.shape)

    clf.fit(X_train, y_train, verbose=False)

    # clf.fit(X_train, y_train,
    #         verbose=False,
    #         early_stopping_rounds=2,
    #         eval_set=[(X_val, y_val)])

    # Performance sur le train
    print('train', accuracy_score(clf.predict(X_train), y_train))

    print(X_val.shape)
    if y_val is not None and len(y_val) == X_val.shape[0]:
        print('test X', X_val.shape)
        print('test y', len(y_val))
        print(accuracy_score(clf.predict(X_val), y_val))

    # Step 2: Tune max_depth and min_child_weight
    # param_test1 = {
    #     'max_depth': range(3, 10, 2),
    #     'min_child_weight': range(1, 6, 2)
    # }
    #
    # param_test3 = {
    #     'gamma': [i / 10.0 for i in range(0, 5)]
    # }
    #
    # param_test4 = {
    #     'subsample': [i / 10.0 for i in range(6, 10)],
    #     'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    # }
    #
    # # On affine la recherche FOR Optimal parameter : {'colsample_bytree': 0.7, 'subsample': 0.9}
    # param_test5 = {
    #     'subsample': [i / 100.0 for i in range(65, 80, 5)],
    #     'colsample_bytree': [i / 100.0 for i in range(85, 100, 5)]
    # }
    #
    # param_test6 = {
    #     'learning_rate': [i / 1000.0 for i in range(5, 20, 2)]
    # }
    #
    # gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.015,
    #                                                 n_estimators=1000,
    #                                                 max_depth=9,
    #                                                 min_child_weight=1,
    #                                                 gamma=0,
    #                                                 subsample=0.7,
    #                                                 colsample_bytree=0.9,
    #                                                 colsample_bylevel = 0.9,
    #                                                 reg_alpha = 0.2,
    #                                                 # objective='binary:logistic',
    #                                                 nthread=4,
    #                                                 scale_pos_weight=1,
    #                                                 seed=27,
    #                                                 n_jobs=-1),
    #                         param_grid=param_test3,
    #                         # scoring='roc_auc',
    #                         scoring='accuracy',
    #                         n_jobs=-1,
    #                         iid=False,
    #                         cv=5)
    #
    # gsearch1.fit(X_train, y_train)
    # # print(gsearch1.grid_scores_)
    # print(gsearch1.best_params_, gsearch1.best_score_)


if __name__ == '__main__':
    pass
    # _Xtrain, _ytrain, y_train, y_test = data_set_load(test_size=0.1,
    #                                                   random_state=42)
    #
    # labels = _ytrain.unique()
    #
    # # Balance distribution -> _Xtrain|_ytrain (overwritten)
    # print_class_balance('Current fold category distribution', _ytrain, labels)
    # _Xtrain, _ytrain = balance_class_by_over_sampling(_Xtrain, _ytrain)
    # print_class_balance('after balanced', _ytrain, labels)
