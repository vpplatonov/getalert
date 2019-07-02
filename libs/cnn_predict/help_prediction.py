import time
import logging

from collections import OrderedDict
from help_detection.helpers import read_file
from .utils import send_on_predict


logger = logging.getLogger('app.py')


async def load_labels(filename):
    """Read in labels, one label per line."""
    with open(filename) as f:
        return [line.rstrip() for line in f.readlines()]


async def get_top_predict(predictions, how_many_labels, labels_list):
    top_pred = []
    for prediction in predictions:
        d = {}
        # Sort to show labels in order of confidence
        top_k = prediction.argsort()[-how_many_labels:][::-1]
        for node_id in top_k:
            human_string = labels_list[node_id]
            score = prediction[node_id]
            d[human_string] = score
        sort_result = OrderedDict(sorted(d.items(), key=lambda t: t[1], reverse=True))
        top_pred.append(sort_result)
    return top_pred


async def label_wav(samples, tf_host, conf):
    """Loads the model and labels, and runs the inference to print predictions."""
    # loading labels from txt file    
    labels_list = await load_labels(conf['labels'])
    # create chunks
    t0 = time.time()
    time_ranges, generated_chunks = await read_file(samples, conf)
    logger.info("Time spent on generation - {} sec".format(time.time() - t0))
    # get a prediction for each chunk
    t0 = time.time()
    predictions = list()
    
    ############# TO DO add in the async function #############
    for chunk in generated_chunks:
        chunk_r = chunk.reshape(1, *chunk.shape, 1)
        # send to the TF serving
        model_pred = await send_on_predict(tf_host, 'help_model', 0, chunk_r)
        # logger.info(model_pred)
        predictions.append(model_pred[0])  
    ###########################################################
    
    logger.info("Time spent on prediction - {} sec".format(time.time() - t0))
    # get the top prediction for each chunk
    top_result = await get_top_predict(predictions, conf['how_many_labels'], labels_list)
    return time_ranges, top_result


async def main(request):
    """Entry point for script."""

    config = request.app['help_config']
    tf_host = request.app['config']['tf_serving_host']
    audio_samples = request['audio']
    # send file in the flow
    return await label_wav(audio_samples, tf_host, config)
