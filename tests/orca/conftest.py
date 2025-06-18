import pytest

from daq2lh5.orca.orca_streamer import OrcaStreamer


@pytest.fixture(scope="module")
def orca_stream(lgnd_test_data):
    orstr = OrcaStreamer()
    orstr.open_stream(
        lgnd_test_data.get_path("orca/fc/l200-p02-r008-phy-20230113T174010Z.orca")
    )
    return orstr


@pytest.fixture(scope="module")
def orca_stream_fcio(lgnd_test_data):
    orstr = OrcaStreamer()
    orstr.open_stream(
        lgnd_test_data.get_path("orca/fcio/l200-p14-r004-cal-20250606T010224Z.orca")
    )
    return orstr


@pytest.fixture(scope="module")
def orca_stream_fcio_swt(lgnd_test_data):
    orstr = OrcaStreamer()
    orstr.open_stream(
        lgnd_test_data.get_path("orca/fcio/l200-p13-r007-aph-20250101T003931Z.orca")
    )
    return orstr
