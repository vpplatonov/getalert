import os
import boto3
import datetime
from collections import namedtuple

from libs.model.feed_model_store import db_save_file_info, FEED_TEST, get_db

# COLLECTION_FILE = 'feed_files'
COLLECTION_FILE = 'fp_sounds'
CLASS_PREDICTED = 'crying_baby'
DB_NAME = 'local'

MIN_IO = 'http://127.0.0.1:9000'
MIN_IO_BUCKET = 'sound.detections'
aws_secret_access_key = "3qHnT7bUaSUIDIBn1bYgG9NZmDqoIThRmFPlqiNk"
aws_access_key_id = "Y82N14S1Q7095ZBWU12L"

MinResource = namedtuple('MinResource', ['service_name', 'endpoint_url', 'aws_secret_access_key', 'aws_access_key_id'])
MinResource.__new__.__defaults__ = ('s3', '', '', '')
MinS3Local = MinResource("s3", MIN_IO, aws_secret_access_key, aws_access_key_id)


def main():

    collection = get_db(db_name=DB_NAME)[COLLECTION_FILE]
    # The name of the service for which a client will be created.
    s3_client = boto3.resource(**MinS3Local._asdict())

    try:
        bucket = s3_client.Bucket(MIN_IO_BUCKET)
        print('delete_bucket', bucket.objects.all().delete())
        print(bucket.delete())
    except:
        pass

    resp = s3_client.create_bucket(Bucket=MIN_IO_BUCKET)
    print('create_bucket', resp)

    load_path_model = os.path.normpath('../cnn_predicted_cry')
    file_list = os.listdir(load_path_model)
    for audio_file in file_list:
        # Save info to MongoDB
        db_save_file_info(collection, FEED_TEST, MIN_IO_BUCKET + '/' + FEED_TEST + '/' + audio_file, CLASS_PREDICTED)
        # Save to MinIO
        # path_to_file = r"C:\Users\User\...\20190514081917099_74163_4.001.wav"
        resp = bucket.upload_file(
            Filename=os.path.join(os.path.abspath(load_path_model), audio_file),
            # Body=path_to_file,
            # ContentType = 'audio/x-wav',
            # Bucket='sound.detections',
            Key=FEED_TEST + '/' + audio_file
        )
        print(resp)


if __name__ == '__main__':
    main()
