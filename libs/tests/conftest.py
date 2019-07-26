import pytest
import os
import boto3
import asyncio
import pymongo

from collections import namedtuple
from pathlib import Path
from libs.predict.feature_engineer import conf_load
from libs.model.xgboost_db_save import MinS3Local
from libs.cnn_predict.config import get_config

# MongoDB param
DB_HOST = 'localhost'
DB_PORT = '27017'
# DB_NAME = 'feed_filter'

CLASS_PREDICTED = 'crying_baby'
# FIXME: use different DB_NAME & MIN_IO_BUCKET
DB_NAME = 'feed_files'
COLLECTION_FILE = 'fp_sounds'

# Minio S3 param
MIN_IO_BUCKET = 'sound.detections'

# File Upload param
load_path_model = os.path.normpath('../getalert/cnn_predicted_cry')
# File Upload param
FileList = namedtuple('FileList', ['path', 'num_files'])
FileLists = [FileList(load_path_model, 12),
             FileList(os.path.normpath('../getalert/cnn_predicted_1_cry'), 36)]


def pytest_addoption(parser):
    """PyTest method for adding custom console parameters"""
    parser.addoption("--additional_value",
                     action="store",
                     default=0,
                     type=int,
                     help="Set additional value for timestamp")


def idparametrize(name, values, fixture=False):
    return pytest.mark.parametrize(name,
                                   values,
                                   ids=list(map(repr, values)),
                                   indirect=fixture)


@pytest.fixture(scope='session')
def case_conf_default():
    conf = get_config()
    yield conf


@pytest.fixture(scope='function')
def case_conf_load():
    dataroot = Path('.')
    conf = conf_load(dataroot)
    yield conf


@pytest.fixture(scope='session')
def _loops():
    _loop = asyncio.get_event_loop()
    yield _loop
    _loop.close()


@pytest.fixture(scope='session')
def s3_client():
    session = boto3.session.Session()
    # The name of the service for which a client will be created.
    s3_client = session.resource(**MinS3Local._asdict())

    return s3_client


@pytest.fixture(scope='session')
def mongo_connect():
    myclient = pymongo.MongoClient("mongodb://{}:{}/".format(DB_HOST, DB_PORT))

    yield myclient

    myclient.close()


@pytest.fixture(scope='session')
def file_list_prep():
    file_list = os.listdir(load_path_model)

    return file_list
