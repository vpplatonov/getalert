import pytest
import numpy as np
import os
import boto3
from botocore.exceptions import ClientError

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

COLLECTION_FILE = 'fp_sounds'
CLASS_PREDICTED = 'crying_baby'
DB_NAME = 'local'

# Minio S3 param
MIN_IO_BUCKET = 'sound.detections'

MIN_IO = 'http://127.0.0.1:9000'
aws_secret_access_key = "3qHnT7bUaSUIDIBn1bYgG9NZmDqoIThRmFPlqiNk"
aws_access_key_id = "Y82N14S1Q7095ZBWU12L"
MinResource = namedtuple('MinResource', ['service_name', 'endpoint_url', 'aws_secret_access_key', 'aws_access_key_id'])
MinResource.__new__.__defaults__ = ('s3', '', '', '')
MinS3Local = MinResource("s3", MIN_IO, aws_secret_access_key, aws_access_key_id)


@pytest.fixture(scope='module')
def s3_client():
    session = boto3.session.Session()
    # The name of the service for which a client will be created.
    s3_client = session.resource(**MinS3Local._asdict())

    return s3_client


# @pytest.fixture(scope='module')
def create_bucket(s3_client):
    resp = s3_client.create_bucket(Bucket=MIN_IO_BUCKET)

    return resp


@pytest.fixture(scope='module')
def mongo_connect():
    myclient = pymongo.MongoClient("mongodb://{}:{}/".format(DB_HOST, DB_PORT))

    return myclient


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
    def test_s3_create_bucket(self, s3_client):
        # Create bucket
        try:
            resp = create_bucket(s3_client)
            print(resp)

            assert resp

        except ClientError as e:
            print(e)
            assert False
            # _bucket = s3_client.Bucket(MIN_IO_BUCKET)
            # _bucket.objects.all().delete()
            # _bucket.delete()

    def test_mongo(self, mongo_connect):
        assert mongo_connect[DB_NAME]
