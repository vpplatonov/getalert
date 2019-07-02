import ujson
import time
import glob
import asyncio
import aiohttp
import logging

import pandas as pd


logger = logging.getLogger('app.py')


async def sender(data):
    t_start = time.time()
    try:
        async with aiohttp.ClientSession(json_serialize=ujson.dumps) as session:
            async with session.get(**data) as response:
                resp = {'status': response.status, 'resp': await response.json(), 'time': time.time() - t_start, 'file_name': data['params']['file_name']}
                return resp
    except (aiohttp.client_exceptions.ServerDisconnectedError,
            aiohttp.client_exceptions.ClientConnectorError) as err:
        logging.exception("Exception occurred")
        return 400, repr(err)


async def main(all_files):

    all_resp = {'True/False': [],
                'all_time': [],
                'file_name': []}

    # strategy 2
    for ind, wav in enumerate(all_files):
        tasks = []
        
        if ind % 100 == 0:
            print(ind)
        urls = [
            # 'http://127.0.0.1:8808/api/v1/is_speech',
            # 'http://127.0.0.1:8808/api/v1/predict/help',
            'http://127.0.0.1:8808/api/v1/predict/crying_baby',
            # 'http://127.0.0.1:8808/api/v1/predict/scream',
        ]
        for url in urls:
            data = {
                'url': url,
                'params': {'file_name': wav},
            }
            tasks.append(sender(data))
            tasks.append(sender(data))
            tasks.append(sender(data))

        resps = await asyncio.gather(*tasks)
        # print(resps)
        for resp in resps:
            all_resp['True/False'].append(resp['resp']['predicted'])
            # all_resp['True/False'].append(resp['resp']['speech_detected'])
            all_resp['all_time'].append(resp['time'])
            all_resp['file_name'].append(resp['file_name'])

        # print(resp)
    return all_resp


fp_paths = [
    r"C:\Users\User\workspace\getalert\cnn_predicted_cry",
    # r"C:\Users\User\workspace\getalert\audio_samples",
]

all_files = [i for p in fp_paths for i in glob.glob(p + '/*.wav')]
print(len(all_files), all_files[:3])

# Async way
t0 = time.time()

# Iter way
t0 = time.time()
all_resp = {'True/False': [], 'all_time': [], 'file_name': []}
loop = asyncio.get_event_loop()

t_100 = time.time()
for ind, file_name in enumerate(all_files[:]):
    t1 = time.time()
    # if ind % 100 == 0:
    #     t_100_2 = time.time()
    #     print(ind, 'time spended -',  t_100_2 - t_100)
    #     t_100 = t_100_2

    data = {
        'url': 'http://127.0.0.1:8808/api/v1/predict/crying_baby',
        'params': {'file_name': file_name},
    }
    resp = loop.run_until_complete(sender(data))
    print(resp)

    all_resp['True/False'].append(resp['resp']['predicted'])
    all_resp['all_time'].append(time.time() - t1)
    all_resp['file_name'].append(file_name)

# print(time.time() - t0)
loop.close()

# df = pd.DataFrame(all_resp)
# # df.to_pickle('test_help_result.npy')
# df.to_pickle('test_cry_result.npy')



