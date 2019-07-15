import pytest
import socket as s
from collections import namedtuple
from .conftest import idparametrize

# FIXME: fixture parametrize use eval() - 'Srv' must be globally seen
Srv = namedtuple('Srv', ['host', 'port'])
REAL_IP = s.gethostbyname(s.gethostname())
ServiceLocations = [Srv('localhost', 8500), Srv(REAL_IP, 8500)]


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


class TestTFServing():

    @idparametrize('Server', ServiceLocations, fixture=True)
    def test_server_connect(self, socket, Server):
        socket.connect(Server.host_port)
        assert socket

    def test_serving_resource(self):
        pass

    def test_serving_resource_validation(self):
        pass
