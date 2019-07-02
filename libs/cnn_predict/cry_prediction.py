import os
# Warning suppression
import warnings
import numpy as np
import asyncio

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from .utils import send_on_predict
from .helpers import read_as_melspectrogram, split_long_data


warnings.simplefilter('ignore')
np.warnings.filterwarnings('ignore')
np.random.seed(1001)


def convert_to_labels(preds, i2c, k=2):
    ans = []
    ids = []
    for p in preds:
        s = np.argsort(p)
        idx = s[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids


def pred_geometric_mean(preds_set):
    predictions = np.ones_like(preds_set[0])
    for preds in preds_set:
        predictions = predictions*preds
    predictions = predictions**(1./len(preds_set))
    return predictions


def predict_generator(conf, X_predict, tf_host, fold):
    aug_datagen = ImageDataGenerator(
        featurewise_center=conf['normalize'] == 'featurewise',
        featurewise_std_normalization=conf['normalize'] == 'featurewise',
    )

    if aug_datagen.featurewise_center:
        aug_datagen.mean, aug_datagen.std = np.mean(X_predict), np.std(X_predict)

    _y = to_categorical(np.ones((len(X_predict))))
    test_generator = aug_datagen.flow(X_predict, _y, batch_size=1, shuffle=False)

    preds = []
    loop = asyncio.get_event_loop()
    ############# TO DO add in the async function #############
    for i in range(len(test_generator)):
        chunk_r = test_generator.next()[0]
        model_pred = loop.run_until_complete(send_on_predict(tf_host, 'cry_model', fold, chunk_r))
        preds.append(model_pred[0])
    ###########################################################

    # preds = model.predict_generator(test_generator, steps=len(test_generator))
    return preds


def file_to_data(conf, audio_samples):
    time_range, data = read_as_melspectrogram(conf, audio_samples)

    X = []
    time_ranges = list()
    for chunk in split_long_data(conf, data, time_range):
        X.append(np.expand_dims(chunk[0], axis=-1))
        time_ranges.append(chunk[-1])
    X = np.array(X)

    return time_ranges, X


def label_wav(audio_samples, tf_host, conf):
    """Loads the model and labels, and runs the inference to print predictions."""
    # read data
    time_ranges, data = file_to_data(conf, audio_samples)

    #### strategy for single model initialization ####
    # prediction = list()
    # for fold in range(5):
    #     # get the fold name
    #     fold_name = os.path.join(conf['path_to_model'], 'best_%d.h5' % fold)
    #     # load weights
    #     model.load_weights(fold_name)
    #     y = await predict_generator(conf, data, conf['model'])
    #     # print(y)
    #     prediction.append([y])

    #### strategy for multiple models initialization ####
    prediction = list()
    for fold in range(5):
        y = predict_generator(conf, data, tf_host, fold)
        prediction.append([y])

    p = pred_geometric_mean(prediction)

    predictions = list()
    for prdct in p:
        # for python 3.5 k=1 urgent
        str_preds, idx = convert_to_labels(prdct, conf['i2c'], k=1)
        for i, str in enumerate(str_preds):
            # work well only in 3.6
            row = dict(zip(str.split(' '), prdct[i][idx[i]]))
            # print(row)
            predictions.append(row)
    return time_ranges, predictions


def main(request):
    """Entry point for script."""
    config = request.app['cry_config']
    tf_host = request.app['config']['tf_serving_host']
    audio_samples = request['audio']
    # send file in the flow
    return label_wav(audio_samples, tf_host, config)
