import numpy as np
import pandas as pd
import asyncio
import botocore
import aiobotocore
from pathlib import Path

import os
import io
import pickle

from model.xgboost_db_save import COLLECTION_FILE, CLASS_PREDICTED, DB_NAME, MIN_IO, aws_secret_access_key, aws_access_key_id
from model.feed_model_store import get_db, FEED_TEST
from predict.feature_engineer import NUM_PCA, MODEL_TYPE, read_audio, get_mfcc_feature, conf_load, FOLDER, audio_load
from model.data_set_train import get_class_id

MIN_IO_BUCKET = 'sound.detections'


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


def main():
    DATAROOT = Path('./../../GetAlertCNN/GetAlertCNN')
    conf = conf_load(DATAROOT, folder=FOLDER)
    print(conf)

    X_train_extra = np.array([])
    y_train_extra = []

    files = get_s3_files()
    loop = asyncio.get_event_loop()

    for i, file in enumerate(files):
        data = loop.run_until_complete(get_s3_file(loop, file))
        xx = audio_load(conf, pathname=data, pydub_read=True)

        class_ids = get_class_id(xx, conf)
        class_ids_keys = []
        for class_id in class_ids:
            class_ids_keys.append(list(class_id.keys())[0])

        if X_train_extra.shape[0] == 0:
            X_train_extra = np.array(xx)
            y_train_extra = np.array(class_ids_keys)
        else:
            X_train_extra = np.concatenate((X_train_extra, np.array(xx)), axis=0)
            y_train_extra = np.concatenate((y_train_extra, np.array(class_ids_keys)), axis=0)

        print(X_train_extra.shape)

    print(y_train_extra)
    print(len(y_train_extra))
    print(len(X_train_extra))


if __name__ == '__main__':
    main()

