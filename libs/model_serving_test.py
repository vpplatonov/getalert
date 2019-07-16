import grpc
import time
import asyncio

import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


async def send_on_predict(hots_port, model_name, version, data):
    """Send array to the Tensorflow Serving.

    Must be wrapped in high level function. @see serving_test.py
        try:
        except grpc.RpcError as e:

    Args:
        hots_port: The host:port string.
        model_name: The model name.
        version: The model version.
        data: The numpy array.

    Returns:
        predictions
    Exception:
        grpc.RpcError
    """

    # create the channel
    channel = grpc.insecure_channel(hots_port)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # create the request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.version.value = version
    # convert the data
    send_data = tf.contrib.util.make_tensor_proto(data.astype(np.float32))
    request.inputs['input_array'].CopyFrom(send_data)
    # send on prediction
    result = stub.Predict(request, 10.0)  # 10 secs timeout

    return np.expand_dims(result.outputs["dense_2/Softmax:0"].float_val, axis=0)


async def main():

    # hots_port = 'modelsserving-engine:8500'
    hots_port = 'localhost:8500'
    # help_model = 'help_model'
    cry_model = 'cry_model'

    all_resps = []

    main_iter = 1
    per_iter = 1

    ## warm up
    # await send_on_predict(hots_port, cry_model, 0,
    #                 np.random.rand(1, 40, 32, 1).astype(np.float32)
    # )

    start_time = time.time()

    tasks = [send_on_predict(hots_port,
                             cry_model,
                             fold,
                             np.random.rand(1, 48, 201, 1).astype(np.float32)
            ) for fold in range(5)]
    print(len(tasks))
    results = await asyncio.gather(*tasks)
    all_resps.append(results)

    end_time = time.time() - start_time
    print('full time spent -', end_time)

    return all_resps


if __name__ == "__main__":

    loop = asyncio.get_event_loop()
    resp = loop.run_until_complete(main())
    print(len(resp))
    np.save('resp.npy', resp)
    loop.close()
