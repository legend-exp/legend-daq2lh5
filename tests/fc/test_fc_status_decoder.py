from multiprocessing.dummy import Array

import lgdo
from fcio import Tags as FCIOTag
import numpy as np
import pytest
from pytest import approx

from daq2lh5.fc.fc_status_decoder import FCStatusDecoder, get_key
from daq2lh5.raw_buffer import RawBuffer


@pytest.fixture(scope="module")
def status_rbkd(fcio_obj):
    decoder = FCStatusDecoder()
    decoder.set_fcio_stream(fcio_obj)

    # get first FCIOStatus record
    nrecords = 1 # first FCIOConfig is decoded automatically
    while fcio_obj.get_record():
        nrecords += 1
        if fcio_obj.tag == FCIOTag.Status:
            break

    assert nrecords == 38

    ncards = decoder.get_max_rows_in_packet()

    print(ncards)

    assert ncards == fcio_obj.status.cards

    # build raw buffer for each channel in the FC trace list
    rbkd = {}
    for i, card_status in enumerate(fcio_obj.status.data):
        key = decoder.key_list[i]
        assert key == get_key(fcio_obj.config.streamid, card_status.reqid)
        rbkd[key] = RawBuffer(lgdo=decoder.make_lgdo(key=key, size=1))
        rbkd[key].fill_safety = 1

    decoder.decode_packet(fcio=fcio_obj, status_rbkd=rbkd, packet_id=38)

    return rbkd


def test_decoding(status_rbkd):
    for _, v in status_rbkd.items():
        assert v.is_full() is True


def test_data_types(status_rbkd):
    for _, v in status_rbkd.items():
        tbl = v.lgdo

        assert isinstance(tbl["packet_id"], lgdo.Array)
        assert isinstance(tbl["status"], lgdo.Array)
        assert isinstance(tbl["fpga_time"], lgdo.Array)
        assert isinstance(tbl["server_time"], lgdo.Array)
        assert isinstance(tbl["fpga_start_time"], lgdo.Array)

        # per card information
        assert isinstance(tbl["id"], lgdo.Array)
        assert isinstance(tbl["eventnumber"], lgdo.Array)
        assert isinstance(tbl["fpga_time_nsec"], lgdo.Array)
        assert isinstance(tbl["n_total_errors"], lgdo.Array)
        assert isinstance(tbl["n_environment_errors"], lgdo.Array)
        assert isinstance(tbl["n_cti_errors"], lgdo.Array)
        assert isinstance(tbl["n_other_errors"], lgdo.ArrayOfEqualSizedArrays)
        assert isinstance(tbl["mb_temps"], lgdo.ArrayOfEqualSizedArrays)
        assert isinstance(tbl["mb_voltages"], lgdo.ArrayOfEqualSizedArrays)
        assert isinstance(tbl["mb_current"], lgdo.Array)
        assert isinstance(tbl["mb_humidity"], lgdo.Array)
        assert isinstance(tbl["adc_temps"], lgdo.VectorOfVectors)
        assert isinstance(tbl["cti_links"], lgdo.VectorOfVectors)
        assert isinstance(tbl["link_states"], lgdo.VectorOfVectors)

def test_values(status_rbkd, fcio_obj):
    for card_data in fcio_obj.status.data:
        key = get_key(fcio_obj.config.streamid, card_data.reqid)
        loc = status_rbkd[key].loc - 1
        tbl = status_rbkd[key].lgdo

        assert status_rbkd[key].fill_safety == 1

        assert tbl["packet_id"].nda[loc] == 38
        assert tbl["status"].nda[loc] == fcio_obj.status.status
        assert tbl["fpga_time"].nda[loc] == approx(fcio_obj.status.fpga_time_sec)
        assert tbl["server_time"].nda[loc] == approx(fcio_obj.status.unix_time_utc_sec)
        assert tbl["fpga_start_time"].nda[loc] == approx(fcio_obj.status.fpga_start_time_sec)

        # per card information
        assert tbl["id"].nda[loc] == card_data.reqid
        assert tbl["eventnumber"].nda[loc] == card_data.eventno
        assert tbl["fpga_time_nsec"].nda[loc] == card_data.fpga_time_nsec
        assert tbl["n_total_errors"].nda[loc] == card_data.totalerrors
        assert tbl["n_environment_errors"].nda[loc] == card_data.enverrors
        assert tbl["n_cti_errors"].nda[loc] == card_data.ctierrors
        assert np.array_equal(tbl["n_other_errors"].nda[loc], card_data.othererrors)
        assert np.array_equal(tbl["mb_temps"].nda[loc], card_data.mainboard_temperatures_mC)
        assert np.array_equal(tbl["mb_voltages"].nda[loc], card_data.mainboard_voltages_mV)
        assert tbl["mb_current"].nda[loc] == card_data.mainboard_current_mA
        assert tbl["mb_humidity"].nda[loc] == card_data.mainboard_humiditiy_permille

        # custom logic for VectorOfVectors
        start = 0 if loc == 0 else tbl["adc_temps"].cumulative_length.nda[loc - 1]
        stop = start + len(card_data.daughterboard_temperatures_mC)
        assert np.array_equal(
            tbl["adc_temps"].flattened_data.nda[start:stop], card_data.daughterboard_temperatures_mC
        )

        start = 0 if loc == 0 else tbl["ct_links"].cumulative_length.nda[loc - 1]
        stop = start + len(card_data.ctilinks)
        assert np.array_equal(
            tbl["cti_links"].flattened_data.nda[start:stop], card_data.ctilinks
        )

        start = 0 if loc == 0 else tbl["link_states"].cumulative_length.nda[loc - 1]
        stop = start + len(card_data.linkstates)
        assert np.array_equal(
            tbl["link_states"].flattened_data.nda[start:stop], card_data.linkstates
        )
