import pytest
import numpy as np
import os
import boto3
from collections import namedtuple
from botocore.exceptions import ClientError
from .conftest import idparametrize

from libs.model.xgboost_db_save import MinS3Local
from libs.model.feed_model_store import db_save_file_info, FEED_TEST
from libs.model.data_set_train import get_s3_files

import pymongo
import datetime
from pathlib import Path
import pickle
from collections import namedtuple

# next line for pickle.load()
from xgboost import XGBClassifier, XGBRegressor
import json

# MongoDB param
DB_HOST = 'localhost'
DB_PORT = '27017'
# DB_NAME = 'feed_filter'

CLASS_PREDICTED = 'crying_baby'
DB_NAME = 'local'
COLLECTION_FILE = 'fp_sounds'

# Minio S3 param
MIN_IO_BUCKET = 'sound.detections'

# File Upload param
load_path_model = os.path.normpath('../cnn_predicted_cry')
FileList = namedtuple('FileList', ['path', 'num_files'])
FileLists = [FileList(load_path_model, 12),
             FileList(os.path.normpath('../cnn_predicted_1_cry'), 36)]


@pytest.fixture(scope='session')
def s3_client():
    session = boto3.session.Session()
    # The name of the service for which a client will be created.
    s3_client = session.resource(**MinS3Local._asdict())

    return s3_client


@pytest.fixture(scope='session')
def mongo_connect():
    myclient = pymongo.MongoClient("mongodb://{}:{}/".format(DB_HOST, DB_PORT))

    return myclient


@pytest.fixture(scope='module')
def file_list_prep():

    file_list = os.listdir(load_path_model)

    return file_list


@pytest.fixture(scope='module')
def file_list_prepare(request):

    class Structure:
        def __init__(self, flist):
            # file_list = flist._asdict()
            self.file_list = os.listdir(flist.path)
            self.num_files = flist.num_files

    return Structure(request.param)


@pytest.fixture(scope='module')
def mongo_prepare(mongo_connect, file_list_prepare):
    myclient = mongo_connect
    db = myclient[DB_NAME]

    # prepare mongo data
    for audio_file in file_list_prepare.file_list:
        # Save info to MongoDB
        db_save_file_info(db[COLLECTION_FILE],
                          FEED_TEST,
                          MIN_IO_BUCKET + '/' + FEED_TEST + '/' + audio_file,
                          CLASS_PREDICTED)

    yield db

    # clear all data in mongo
    db.drop_collection(COLLECTION_FILE)


@pytest.fixture(scope='module')
def create_bucket(s3_client):
    try:
        resp = s3_client.create_bucket(Bucket=MIN_IO_BUCKET)
    except ClientError as e:
        print(e)
        resp = s3_client.Bucket(MIN_IO_BUCKET)

    yield resp

    # delete bucket on Minio S3
    _bucket = s3_client.Bucket(MIN_IO_BUCKET)
    # _bucket.objects.all().delete()
    _bucket.delete()


@pytest.fixture(scope='module')
def prepare_bucket(create_bucket):

    yield

    # clear bucket on Minio S3
    # _bucket = s3_client.Bucket(MIN_IO_BUCKET)
    create_bucket.objects.all().delete()
    # create_bucket.delete()


class TestMongoS3():

    def test_S3_client(self):
        session = boto3.session.Session()
        s3_client = session.resource(**MinS3Local._asdict())

        assert s3_client

    def test_s3_bucket(self, s3_client):
        bucket = s3_client.Bucket(MIN_IO_BUCKET)

        # Output the bucket names
        assert bucket

    @pytest.mark.xfail()
    def test_s3_create_bucket(self, create_bucket):
        # Create bucket
        try:
            resp = create_bucket
            print(type(resp))

            assert resp

        except ClientError as e:
            print(e)
            print(type(create_bucket))
            assert False

    def test_mongo(self, mongo_connect):
        assert mongo_connect[DB_NAME]

    @idparametrize('file_list_prepare', FileLists, fixture=True)
    def test_file_list_prepare(self, file_list_prepare):
        assert len(file_list_prepare.file_list) == file_list_prepare.num_files

    @idparametrize('file_list_prepare', FileLists, fixture=True)
    def test_mongo_data(self, mongo_prepare, file_list_prepare):
        collection = mongo_prepare[COLLECTION_FILE]
        files = get_s3_files(collection,
                             feed_id=FEED_TEST,
                             class_predicted=CLASS_PREDICTED)
        assert files.count() == file_list_prepare.num_files

