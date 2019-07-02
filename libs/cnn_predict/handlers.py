import os
import logging
import ujson
import time

from aiohttp import web
from .utils import if_speech_exist, file_to_array, predict_category
from .help_prediction import main as help_main
from .cry_prediction import main as cry_main

logger = logging.getLogger('app.py')


async def is_speech(request):
    # send to the VAD
    speech_detected = await if_speech_exist(request['audio'], request.app['config'])
    # return True/False
    response = {'speech_detected': speech_detected}
    return web.json_response(response, status=200, dumps=ujson.dumps)


async def predict_help(request):
    t_start = time.time()
    # get the file_name
    file_name = request.query['file_name']
    # return start/stop time ranges
    time_ranges, preds = await help_main(request)
    # get category from url
    category = request.path.split('/')[-1]
    # voting strategy
    pred = await predict_category(preds, time_ranges,
                                  category=category,
                                  strategy='Only_first', threshold=0.65)
    logger.info("category - {} | file_name - {} | predicted - {} |\nTotal time spent - {} in sec".format(
        category, file_name, 1 if pred else 0, time.time() - t_start))
    # saving the result to the file
    resp = {'predicted': bool(pred), 
            'time_ranges': pred
    }
    return web.json_response(resp, status=200, dumps=ujson.dumps)


async def predict_cry_or_scream(request):
    t_start = time.time()
    # get the file_name
    file_name = request.query['file_name']
    # return start/stop time ranges
    time_ranges, preds = await cry_main(request)
    # get category from url
    category = request.path.split('/')[-1]
    # voting strategy
    pred = await predict_category(preds, time_ranges,
                                  category=category,
                                  strategy='Once',
                                  threshold=0.55 if category == 'crying_baby' else 0.35)
    logger.info("category - {} | file_name - {} | predicted - {} |\nTotal time spent - {} in sec".format(
        category, file_name, 1 if pred else 0, time.time() - t_start))
    # saving the result to the file
    resp = {'predicted': bool(pred), 
            'time_ranges': pred
    }
    return web.json_response(resp, status=200, dumps=ujson.dumps)
