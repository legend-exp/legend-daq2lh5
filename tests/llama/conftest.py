import pytest


# lgnd_test_data (LegendTestData) from fixture in root conftest.py
@pytest.fixture(scope="module")
def test_data_path(lgnd_test_data):
    return lgnd_test_data.get_path("llamaDAQ/20241218-150158-pulser.bin")
