import numpy as np
from imblearn.over_sampling import RandomOverSampler
# from .data_set_train import data_set_load
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pathlib import Path

#trained parameters const
MAX_DEPTH = 'max_depth'
MIN_CHILD_WEIGHT = 'min_child_weight'
GAMMA = 'gamma'
SUBSAMPLE = 'subsample'
COLSAMPLE_BYTREE = 'colsample_bytree'
LEARNING_RATE = 'learning_rate'


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
