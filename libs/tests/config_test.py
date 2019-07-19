import time
import pytest
from pathlib import Path
from collections import namedtuple
from .conftest import idparametrize

from libs.predict.feature_engineer import conf_load

conf_fields = ['audio_split', 'dims', 'learning_rate', 'folder', 'hop_length', 'samples', 'sampling_rate', 'duration',
               'normalize', 'n_mels', 'n_mfcc', 'fmin', 'n_fft', 'fmax']

config_default_fields = ['vad_mode', 'max_possible_amplitude', 'vad_window_duration',
                         'sample_width', 'headroom', 'filter_flag', 'path_to_model', 'i2c', 'channels', 'labels']

Conf = namedtuple('Conf', conf_fields)
Config = namedtuple('Config', conf_fields + config_default_fields)

ModelParam = namedtuple('ModelParam',
                        ['model', 'duration', 'sampling_rate'])
Models = [ModelParam('X2M16', 2, 16000),
          ModelParam('X3M16', 3, 16000),
          ModelParam('X5M16', 5, 16000)]
CNN_PATH = Path('../../GetAlertCNN/GetAlertCNN')


@pytest.fixture(scope='class', autouse=True)
def suite_data():
    print("\n> Suite setup")
    yield
    print("> Suite teardown")


@pytest.fixture(scope='function')
def case_data():
    print("   > Case setup")
    yield time.time()
    print("\n   > Case teardown")


@pytest.fixture(scope='module')
def config_default(request):

    class Structure:
        def __init__(self, conf):
            # FIXME: ../../GetAlernCNN/GetAlernCNN
            self.conf = conf_load(CNN_PATH, folder=conf.model)
            self.param = conf

    return Structure(request.param)


class TestSuite():

    @pytest.mark.xfail()
    def test_case_1(self, case_data):
        time = int(case_data)
        print("      > Received from fixture timestamp is: {}".format(time))
        assert(time % 2 == 0)

    @pytest.mark.xfail()
    def test_case_2(self, case_data, additional_value):
        time = int(case_data)
        parameter = additional_value
        print("      > Received from fixture timestamp is: {}".format(time))
        print("      > Received from comandline parameter value is: {}".format(parameter))
        time += parameter
        assert(time % 2 == 0)

    def test_conf_load(self, case_conf_load):
        assert case_conf_load
        assert Conf(**case_conf_load)
        assert type(case_conf_load) == dict
        assert case_conf_load['sampling_rate'] == 16000
        assert case_conf_load['duration'] == 3    \

    @idparametrize('config_default', Models, fixture=True)
    def test_conf_load_param(self, config_default):
        assert config_default.conf
        assert Conf(**config_default.conf)
        assert type(config_default.conf) == dict
        assert config_default.conf['sampling_rate'] == config_default.param.sampling_rate
        assert config_default.conf['duration'] == config_default.param.duration

    def test_conf_default(self, case_conf_default):
        assert Config(**case_conf_default)
        assert type(case_conf_default) == dict
        assert case_conf_default['sampling_rate'] == 16000
        assert case_conf_default['duration'] == 3


@pytest.fixture(scope="session")
def additional_value(request):
    """Handler for --additional_value parameter"""
    return request.config.getoption("--additional_value", default=0)
