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
    assert dec_vals_0["waveform_windowed"]["wf_len"] == 2000
    assert dec_vals_0["waveform_presummed"]["wf_len"] == 10000
    dec_vals_4 = evt_dec.get_decoded_values(4)
    assert dec_vals_4["waveform_windowed"]["wf_len"] == 2000
    assert dec_vals_4["waveform_presummed"]["wf_len"] == 500


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
    assert tbl["fadc_channel_id"].nda[ii] == 0
    assert tbl["packet_id"].nda[ii] == 1
    assert tbl["time_since_run_start"].nda[ii] == pytest.approx(0.00303012)
    assert tbl["wf_max_sample_value"].nda[ii] == 9454
    assert tbl["wf_max_sample_idx"].nda[ii] == 1968
    assert tbl["info_bits"].nda[ii] == 0
    assert tbl["cumsum_1"].nda[ii] == 7826
    assert tbl["cumsum_2"].nda[ii] == 7826
    assert tbl["cumsum_3"].nda[ii] == 7826
    assert tbl["cumsum_4"].nda[ii] == 7826
    assert tbl["cumsum_5"].nda[ii] == 7826
    assert tbl["cumsum_6"].nda[ii] == 7826
    assert tbl["cumsum_7"].nda[ii] == 7826
    assert tbl["cumsum_8"].nda[ii] == 7826
    assert (
        tbl["waveform_windowed"]["dt"].nda[ii] > 3.999
        and tbl["waveform_windowed"]["dt"].nda[ii] < 4.001
    )
    assert (
        tbl["waveform_presummed"]["dt"].nda[ii] > 15.999
        and tbl["waveform_presummed"]["dt"].nda[ii] < 16.001
    )
    assert (
        tbl["waveform_windowed"]["t0"].nda[ii] > -4000.1
        and tbl["waveform_windowed"]["t0"].nda[ii] < -3999.9
    )
    assert (
        tbl["waveform_presummed"]["t0"].nda[ii] > -8000.1
        and tbl["waveform_presummed"]["t0"].nda[ii] < -7999.9
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
    assert tbl["fadc_channel_id"].nda[ii] == 4
    assert tbl["packet_id"].nda[ii] == 10
    assert tbl["time_since_run_start"].nda[ii] == pytest.approx(0.00303012)
    assert tbl["wf_max_sample_value"].nda[ii] == 7923
    assert tbl["wf_max_sample_idx"].nda[ii] == 371
    assert tbl["info_bits"].nda[ii] == 0
    assert tbl["cumsum_1"].nda[ii] == 7912
    assert tbl["cumsum_2"].nda[ii] == 7912
    assert tbl["cumsum_3"].nda[ii] == 7912
    assert tbl["cumsum_4"].nda[ii] == 7912
    assert tbl["cumsum_5"].nda[ii] == 7912
    assert tbl["cumsum_6"].nda[ii] == 7912
    assert tbl["cumsum_7"].nda[ii] == 7912
    assert tbl["cumsum_8"].nda[ii] == 7912
    assert (
        tbl["waveform_windowed"]["dt"].nda[ii] > 3.999
        and tbl["waveform_windowed"]["dt"].nda[ii] < 4.001
    )
    assert (
        tbl["waveform_presummed"]["dt"].nda[ii] > 31.999
        and tbl["waveform_presummed"]["dt"].nda[ii] < 32.001
    )
    assert (
        tbl["waveform_windowed"]["t0"].nda[ii] > -4000.1
        and tbl["waveform_windowed"]["t0"].nda[ii] < -3999.9
    )
    assert (
        tbl["waveform_presummed"]["t0"].nda[ii] > -4000.1
        and tbl["waveform_presummed"]["t0"].nda[ii] < -3999.9
    )


def test_event_count(open_stream):
    evt_rbkd = open_stream.event_rbkd
    while True:
        good_packet = open_stream.read_packet()
        if not good_packet:
            break
    assert evt_rbkd[0].loc == 37
    assert evt_rbkd[4].loc == 37
