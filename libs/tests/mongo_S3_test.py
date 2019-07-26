import pytest
import numpy as np
import os
import boto3
from collections import namedtuple
from botocore.exceptions import ClientError
from .conftest import idparametrize, load_path_model, FileList, FileLists
import collections  # From Python standard library.
import json
import bson
from bson.codec_options import CodecOptions
from bson import json_util

from libs.model.xgboost_db_save import MinS3Local
from libs.model.feed_model_store import db_save_file_info, FEED_TEST
from libs.model.data_set_train import get_s3_files, get_s3_file, get_s3_object
from libs.predict.feature_engineer import get_samples
from .conftest import DB_HOST, DB_PORT, DB_NAME, COLLECTION_FILE, MIN_IO_BUCKET, CLASS_PREDICTED

import pymongo
import datetime
from pathlib import Path
import pickle
from collections import namedtuple

# next line for pickle.load()
from xgboost import XGBClassifier, XGBRegressor


@pytest.fixture(scope='module')
def mongo_bson_prepare():
    s = '{"_id": {"$oid": "4edebd262ae5e93b41000000"}}'
    u = json.loads(s, object_hook=json_util.object_hook)
    return u


@pytest.fixture(scope='module')
def mongo_files_read(mongo_prepare, file_list_prepare):
    collection = mongo_prepare[COLLECTION_FILE]
    files = get_s3_files(collection,
                         feed_id=FEED_TEST,
                         class_predicted=CLASS_PREDICTED)

    return files


@pytest.fixture(scope='module')
def file_list_prepare(request):
    """
    Use for parametrized test with decorator
    @idparametrize(..., ..., fixture=True)

    :param request:
    :return:
    """
    class Structure:
        def __init__(self, flist):
            # file_list = flist._asdict()
            self.file_list = os.listdir(flist.path)
            self.num_files = flist.num_files
            self.path = flist.path

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
    _bucket.delete()


@pytest.fixture(scope='module')
def prepare_bucket(create_bucket, file_list_prepare):
    """
    Move all files to S3 bucket
    :param create_bucket:
    :return:
    """
    for audio_file in file_list_prepare.file_list:
        # Save to MinIO
        create_bucket.upload_file(
            Filename=os.path.join(os.path.abspath(file_list_prepare.path), audio_file),
            Key=FEED_TEST + '/' + audio_file
        )

    yield

    # clear bucket on Minio S3
    create_bucket.objects.all().delete()


class TestMongoS3():

    def test_bson_decoder(self, mongo_bson_prepare):
        s = json.dumps(mongo_bson_prepare, default=json_util.default)
        # print(mongo_bson_prepare)
        u = bson.BSON.encode(mongo_bson_prepare)
        # print(type(u))

        assert isinstance(u, type(bson.BSON()))
        assert isinstance(mongo_bson_prepare, dict)
        assert isinstance(s, str)

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
        """

        :param create_bucket:
        :return:
        """
        # Create bucket
        try:
            resp = create_bucket
            print(type(resp))

            assert resp

        except ClientError as e:
            print(e)
            print(type(create_bucket))
            assert False

    @idparametrize('file_list_prepare', FileLists, fixture=True)
    def test_prepare_bucket(self, file_list_prepare, prepare_bucket, s3_client, _loops, case_conf_load):
        for audio_file in file_list_prepare.file_list:
            # data = get_s3_object(s3_client, [MIN_IO_BUCKET, FEED_TEST, audio_file])
            # print(type(data))
            # assert data

            file = _loops.run_until_complete(get_s3_file(_loops, {'filename': '/'.join([MIN_IO_BUCKET, FEED_TEST, audio_file])}))
            samples = get_samples(case_conf_load, file, pydub_read=True)

            assert isinstance(samples, type(np.ndarray([])))
            assert file

    def test_file_list(self, file_list_prep):
        assert len(file_list_prep) == 12

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

    @idparametrize('file_list_prepare', FileLists, fixture=True)
    def test_mongo_read_decode(self, mongo_files_read, file_list_prepare):
        files = mongo_files_read
        # print(files[0])
        assert isinstance(files[0], dict)
        assert files.count() == file_list_prepare.num_files
