import lgdo
import pytest

from daq2lh5.llama.llama_event_decoder import LLAMAEventDecoder, check_dict_spec_equal
from daq2lh5.llama.llama_streamer import LLAMAStreamer


def test_check_dict_spec_equal():
    d1 = {"X": "1", "Y": "2", "Z": "3"}
    d2 = {"X": "2", "Y": "2", "Z": "3"}
    assert check_dict_spec_equal(d1, d2, ["Y", "Z"])
    assert not check_dict_spec_equal(d1, d2, ["X", "Y"])


@pytest.fixture(scope="module")
def open_stream(test_data_path):
    streamer = LLAMAStreamer()
    streamer.open_stream(test_data_path)
    yield streamer
    streamer.close_stream()


def test_get_key_lists(open_stream):
    evt_dec: LLAMAEventDecoder = open_stream.event_decoder
    assert evt_dec.get_key_lists() == [[0], [4]]


def test_get_decoded_values(open_stream):
    evt_dec: LLAMAEventDecoder = open_stream.event_decoder
    dec_vals_0 = evt_dec.get_decoded_values(0)
    assert dec_vals_0["waveform"]["wf_len"] == 2000
    assert dec_vals_0["avgwaveform"]["wf_len"] == 10000
    dec_vals_4 = evt_dec.get_decoded_values(4)
    assert dec_vals_4["waveform"]["wf_len"] == 2000
    assert dec_vals_4["avgwaveform"]["wf_len"] == 500


def test_first_packet(open_stream):
    good_packet = open_stream.read_packet()
    assert good_packet
    evt_dec: LLAMAEventDecoder = open_stream.event_decoder
    assert evt_dec is not None
    evt_rbkd = open_stream.event_rbkd
    tbl = evt_rbkd[0].lgdo
    assert isinstance(tbl, lgdo.Table)
    ii = evt_rbkd[0].loc
    assert ii == 1
    ii = ii - 1  # use the last written entry (which is the only one, actually)
    assert tbl["fch_id"].nda[ii] == 0
    assert tbl["packet_id"].nda[ii] == 1
    assert tbl["timestamp"].nda[ii] == 757530
    assert tbl["peakHighValue"].nda[ii] == 9454
    assert tbl["peakHighIndex"].nda[ii] == 1968
    assert tbl["information"].nda[ii] == 0
    assert tbl["accSum1"].nda[ii] == 7826
    assert tbl["accSum2"].nda[ii] == 7826
    assert tbl["accSum3"].nda[ii] == 7826
    assert tbl["accSum4"].nda[ii] == 7826
    assert tbl["accSum5"].nda[ii] == 7826
    assert tbl["accSum6"].nda[ii] == 7826
    assert tbl["accSum7"].nda[ii] == 7826
    assert tbl["accSum8"].nda[ii] == 7826
    assert (
        tbl["waveform"]["dt"].nda[ii] > 3.999 and tbl["waveform"]["dt"].nda[ii] < 4.001
    )
    assert (
        tbl["avgwaveform"]["dt"].nda[ii] > 15.999
        and tbl["avgwaveform"]["dt"].nda[ii] < 16.001
    )
    assert (
        tbl["waveform"]["t0"].nda[ii] > -4000.1
        and tbl["waveform"]["t0"].nda[ii] < -3999.9
    )
    assert (
        tbl["avgwaveform"]["t0"].nda[ii] > -8000.1
        and tbl["avgwaveform"]["t0"].nda[ii] < -7999.9
    )


def test_first_packet_ch4(open_stream):
    evt_rbkd = open_stream.event_rbkd
    while True:
        good_packet = open_stream.read_packet()
        if not good_packet:
            break
    tbl = evt_rbkd[4].lgdo
    assert evt_rbkd[4].loc > 0, "Not a single event of channel 4"
    ii = 0
    assert tbl["fch_id"].nda[ii] == 4
    assert tbl["packet_id"].nda[ii] == 10
    assert tbl["timestamp"].nda[ii] == 757530
    assert tbl["peakHighValue"].nda[ii] == 7923
    assert tbl["peakHighIndex"].nda[ii] == 371
    assert tbl["information"].nda[ii] == 0
    assert tbl["accSum1"].nda[ii] == 7912
    assert tbl["accSum2"].nda[ii] == 7912
    assert tbl["accSum3"].nda[ii] == 7912
    assert tbl["accSum4"].nda[ii] == 7912
    assert tbl["accSum5"].nda[ii] == 7912
    assert tbl["accSum6"].nda[ii] == 7912
    assert tbl["accSum7"].nda[ii] == 7912
    assert tbl["accSum8"].nda[ii] == 7912
    assert (
        tbl["waveform"]["dt"].nda[ii] > 3.999 and tbl["waveform"]["dt"].nda[ii] < 4.001
    )
    assert (
        tbl["avgwaveform"]["dt"].nda[ii] > 31.999
        and tbl["avgwaveform"]["dt"].nda[ii] < 32.001
    )
    assert (
        tbl["waveform"]["t0"].nda[ii] > -4000.1
        and tbl["waveform"]["t0"].nda[ii] < -3999.9
    )
    assert (
        tbl["avgwaveform"]["t0"].nda[ii] > -4000.1
        and tbl["avgwaveform"]["t0"].nda[ii] < -3999.9
    )


def test_event_count(open_stream):
    evt_rbkd = open_stream.event_rbkd
    while True:
        good_packet = open_stream.read_packet()
        if not good_packet:
            break
    assert evt_rbkd[0].loc == 37
    assert evt_rbkd[4].loc == 37
