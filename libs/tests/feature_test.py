import pytest

# keyword expressions
# # Run all tests with some string ‘validate’ in the name
# pytest -k “validate”

# # Exclude tests with ‘db’ in name but include 'validate'
# pytest -k “validate and not db”

# #Run all test files inside a folder demo_tests
# pytest demo_tests/

# # Run a single method test_method of a test class TestClassDemo
# pytest demo_tests/test_example.py::TestClassDemo::test_method

# # Run a single test class named TestClassDemo
# pytest demo_tests/test_example.py::TestClassDemo

# # Run a single test function named test_sum
# pytest demo_tests/test_example.py::test_sum

# # Run tests in verbose mode:
# pytest -v demo_tests/

# # Run tests including print statements:
# pytest -s demo_tests/

# # Only run tests that failed during the last run
# pytest — lf


@pytest.fixture(scope='session')
def get_sum_test_data_fx():
    return get_sum_test_data()


@pytest.fixture(autouse=True)
def setup_and_teardown():
    print('\nFetching data from db')
    yield
    print('\nSaving test run data in db')


@pytest.mark.slow
def test_sum_fx(get_sum_test_data_fx):
    """
    to run only slow marked tests inside file use
    > pytest <this_test_example>.py -m slow

    :param get_sum_test_data_fx:
    :return:
    """
    for data in get_sum_test_data_fx:
        num1 = data[0]
        num2 = data[1]
        expected = data[2]
        assert summ(num1, num2) == expected


def get_sum_test_data():
    return [(3, 5, 8), (-2, -2, -4), (-1, 5, 4), (3, -5, -2), (0, 5, 5)]


@pytest.mark.parametrize('num1, num2, expected', get_sum_test_data())
def test_sum(num1, num2, expected):
    # make sure to start function name with test
    assert summ(num1, num2) == expected


def test_sum_output_type():
    assert type(summ(1, 2)) is int


def summ(num1, num2):
    """It returns sum of two numbers"""
    return num1 + num2
