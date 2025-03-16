import fcio
import pytest

from daq2lh5.fc.fc_config_decoder import FCConfigDecoder


@pytest.fixture(scope="module")
def fcio_obj(lgnd_test_data):
    return fcio.fcio_open(
        lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio")
    )


@pytest.fixture(scope="module")
def fcio_config(fcio_obj):
    decoder = FCConfigDecoder()
    return decoder.decode_config(fcio_obj)
