import pymongo
import datetime
import numpy as np
from pathlib import Path
import os
import pickle

# next line for pickle.load()
from xgboost import XGBClassifier, XGBRegressor
import json

DB_HOST = 'localhost'
DB_PORT = '27017'
DB_NAME = 'feed_filter'
COLLECTION_MODEL = 'feed_model'
FEED_TEST = '1234-1234513456-234234-sdfg-4354'
CONF_ROOT = 'XGBoost3'
FEED_MODEL = {'feed_id': FEED_TEST,
              'model_type': 'XGBoost',
              'model': '',  # save here model.pkl
              'labels': [],  # to_labels.npy
              'parameters': {},  # save here conf.npy
              'timestamp': datetime.datetime.strptime("2019-06-01T10:59:59.000Z", "%Y-%m-%dT%H:%M:%S.000Z")
              # 'timestamp': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")
              }

DATAROOT = Path('.')


def db_create_fid_filter():

    # try:
    #     db = pymongo.database.Database(myclient, DB_NAME)
    # except pymongo.errors.InvalidName as e:
    #     print(e.message)

    db = get_db(db_name=DB_NAME)
    db.drop_collection(COLLECTION_MODEL)

    # MongoDB 4.0
    # validator = {
    #     '$jsonSchema': {
    #         'required': ['feed_id', 'parameters', 'timestamp'],
    #         'properties': {
    #             'model': {
    #                'bsonType': "binData",
    #                'description': "must be a BLOB and is required"
    #             },
    #             'timestamp': {
    #                 'bsonType': "timestamp",
    #             }
    #         }
    #     }
    # }

    # MongoDB 3.4
    options = {
        'validator': {
            'model': {'$exists': True, '$type': "binData"},
            'timestamp': {'$type': "timestamp"}
        },
        'validationAction': "warn"
    }
    db.create_collection(COLLECTION_MODEL, **options)
    collection = db[COLLECTION_MODEL]
    collection.create_index([('feed_id', pymongo.ASCENDING)], unique=True)

    return db


def db_save_file_info(collection, feed_id, file_name, class_predicted, status=0, filter_class='domestic'):
    collection.insert_one({'feed_id': feed_id,
                           'filename': file_name,
                           'class_predicted': class_predicted,
                           'status': status,
                           'filter_class': filter_class,
                           'timestamp': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")
                           })


def db_save_model(collection, model, parameters={}, feed_id=FEED_TEST, model_type='XGBoost', labels=[], class_name='crying_baby'):
    collection.insert_one({'feed_id': feed_id,
                           'model_type': model_type,
                           'model': model,  # save here model.pkl
                           'labels': labels,  # to_labels.npy
                           'parameters': parameters,  # save here conf.npy,
                           'class': class_name,
                           'timestamp': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")
                           })


def db_update_model(collection, model, feed_id=FEED_TEST):
    collection.update_one(
        {'feed_id': feed_id},
        {'$set': {'model': model},
         '$currentDate': {'timestamp': {'$type': "timestamp"}},
         '$setOnInsert': {'feed_id': feed_id}
         }
    )


def db_load_model(collection, feed_id=FEED_TEST):
    print("feed Id", feed_id)
    cursor = collection.find_one({'feed_id': feed_id}, {"model": 1, "labels": 1, 'parameters': 1, "_id": 0})

    model = cursor['model']
    labels = json.loads(cursor['labels'])
    model = pickle.loads(model)

    return model, labels


def test_db():
    db = get_db(db_name='local')
    mydoc = db.startup_log.find({}, {'buildinfo': 1, "_id": 0})

    for x in mydoc:
        print(x)


def get_db(db_name='local'):
    print("mongodb://{}:{}/".format(DB_HOST, DB_PORT))
    myclient = pymongo.MongoClient("mongodb://{}:{}/".format(DB_HOST, DB_PORT))

    return myclient[db_name]


def save_model_to_db(model, feed_id, class_name='crying_baby'):
    conf = np.load(os.path.join('../../GetAlertCNN/GetAlertCNN/{}'.format(CONF_ROOT), 'conf.npy'))
    labels = np.load(os.path.join('../../output/dataset', 'to_labels.npy'))
    collection = get_db(db_name=DB_NAME)[COLLECTION_MODEL]

    labels = json.dumps(labels.tolist())
    conf = json.dumps(conf.tolist())

    # FID ID for test
    db_save_model(collection, model, labels=labels, parameters=conf, feed_id=feed_id, class_name=class_name)


def main():
    load_path_model = '../../output/model'
    with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
        model = pickle.load(fp)

    model = pickle.dumps(model)

    save_model_to_db(model, FEED_TEST, class_name='help')


if __name__ == '__main__':
    main()
