import lgdo
import numpy as np


def test_decoding(fcio_config):
    pass


def test_data_types(fcio_config):
    assert isinstance(fcio_config, lgdo.Struct)


def test_values(fcio_config):
    expected_dict = {
        "nsamples": 6000,
        "nadcs": 6,
        "ntriggers": 0,
        "streamid": 0,
        "adcbits": 16,
        "sumlength": 1,
        "blprecision": 1,
        "mastercards": 1,
        "triggercards": 0,
        "adccards": 1,
        "gps": 0,
    }

    for k, v in expected_dict.items():
        assert fcio_config[k].value == v

    assert np.array_equal(
        fcio_config["tracemap"].nda,
        np.array(
            [34603008, 34603009, 34603010, 34603011, 34603012, 34603013], dtype="uint32"
        ),
    )
