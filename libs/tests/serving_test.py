import pytest
import socket as s
import numpy as np
import asyncio
import grpc

from collections import namedtuple
from .conftest import idparametrize
from libs.model_serving_test import send_on_predict


# FIXME: fixture parametrize use eval() - 'Srv' must be globally seen
Srv = namedtuple('Srv', ['host', 'port'])
REAL_IP = s.gethostbyname(s.gethostname())
ServiceLocations = [Srv('localhost', 8500), Srv(REAL_IP, 8500)]
cry_model = 'cry_model'


@pytest.yield_fixture
def socket():
    _socket = s.socket(s.AF_INET, s.SOCK_STREAM)
    yield _socket
    _socket.close()


@pytest.fixture(scope='module')
def Server(request):

    class Structure:
        def __init__(self, srv):
            self.srv = srv
            self.host_port = self.srv.host, self.srv.port

        @property
        def uri(self):
            return 'http://{host}:{port}/'.format(**self.srv._asdict())

    return Structure(request.param)


async def get_serving_resource(hots_port, size=[1, 48, 201, 1]):
    np.random.seed(1001)
    all_resps = []
    # print(np.random.rand(1, 48, 201, 1))
    tasks = [send_on_predict(hots_port,
                             cry_model,
                             fold,
                             np.random.rand(*size).astype(np.float32)
            ) for fold in range(5)]

    results = await asyncio.gather(*tasks)
    all_resps.append(results)

    return all_resps


class TestTFServing():

    @idparametrize('Server', ServiceLocations, fixture=True)
    def test_server_connect(self, socket, Server):
        socket.connect(Server.host_port)
        assert socket

    @idparametrize('Server', ServiceLocations, fixture=True)
    def test_serving_resource(self, _loops, Server):
        resp = _loops.run_until_complete(get_serving_resource('{}:{}'.format(Server.host_port[0], Server.host_port[1]),
                                                              size=[1, 48, 201, 1]))
        # print(resp)
        for rs in resp[0]:
            assert type(rs) is np.ndarray
            # assert type(rs[0]) is np.ndarray
            # assert type(rs[1]) is float

    @idparametrize('Server', ServiceLocations, fixture=True)
    def test_serving_resource_validation(self, _loops, Server):
        try:
            _loops.run_until_complete(get_serving_resource('{}:{}'.format(Server.host_port[0], Server.host_port[1]),
                                                           size=[1, 1, 1, 1]))
        except grpc.RpcError as e:
            # print(e.details())
            status_code = e.code()
            # status_code.name
            # status_code.value
            # print(e)
            # want to do some specific action based on the error?
            assert grpc.StatusCode.INVALID_ARGUMENT == status_code
