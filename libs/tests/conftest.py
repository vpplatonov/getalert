import pytest


def pytest_addoption(parser):
    """PyTest method for adding custom console parameters"""
    parser.addoption("--additional_value",
                     action="store",
                     default=0,
                     type=int,
                     help="Set additional value for timestamp")


def idparametrize(name, values, fixture=False):
    return pytest.mark.parametrize(name,
                                   values,
                                   ids=list(map(repr, values)),
                                   indirect=fixture)
