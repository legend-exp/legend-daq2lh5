import lgdo
import numpy as np
import pytest
from fcio import Tags as FCIOTag

from daq2lh5.fc.fc_event_decoder import FCEventDecoder, get_key
from daq2lh5.raw_buffer import RawBuffer


@pytest.fixture(scope="module")
def event_rbkd(fcio_obj):
    decoder = FCEventDecoder()
    decoder.set_fcio_stream(fcio_obj)

    # get just one record and check if it's a (sparse) event
    assert fcio_obj.get_record()
    assert fcio_obj.tag == FCIOTag.Event or fcio_obj.tag == FCIOTag.SparseEvent

    # build raw buffer for each channel in the FC trace list
    rbkd = {}
    for i in fcio_obj.event.trace_list:
        key = decoder.key_list[i]
        rbkd[key] = RawBuffer(lgdo=decoder.make_lgdo(key=key, size=1))
        rbkd[key].fill_safety = 1

    # decode packet into the lgdo's and check if the buffer is full
    assert decoder.decode_packet(fcio=fcio_obj, evt_rbkd=rbkd, packet_id=69) is True

    # # check compression settings (here before any LH5Store.write() call
    # assert "compression" in rbkd[0].lgdo["packet_id"].attrs
    # assert "compression" in rbkd[0].lgdo["waveform"].values.attrs

    return rbkd


def test_decoding_and_compression_attrs(event_rbkd):
    assert event_rbkd != {}


def test_data_types(event_rbkd):
    for _, v in event_rbkd.items():
        tbl = v.lgdo
        assert isinstance(tbl, lgdo.Struct)
        assert isinstance(tbl["packet_id"], lgdo.Array)
        assert isinstance(tbl["eventnumber"], lgdo.Array)
        assert isinstance(tbl["timestamp"], lgdo.Array)
        assert isinstance(tbl["runtime"], lgdo.Array)
        assert isinstance(tbl["lifetime"], lgdo.Array)
        assert isinstance(tbl["deadtime"], lgdo.Array)
        assert isinstance(tbl["numtraces"], lgdo.Array)
        assert isinstance(tbl["baseline"], lgdo.Array)
        assert isinstance(tbl["daqenergy"], lgdo.Array)
        assert isinstance(tbl["channel"], lgdo.Array)
        assert isinstance(tbl["ts_pps"], lgdo.Array)
        assert isinstance(tbl["ts_ticks"], lgdo.Array)
        assert isinstance(tbl["ts_maxticks"], lgdo.Array)
        assert isinstance(tbl["mu_offset_sec"], lgdo.Array)
        assert isinstance(tbl["mu_offset_usec"], lgdo.Array)
        assert isinstance(tbl["to_master_sec"], lgdo.Array)
        assert isinstance(tbl["delta_mu_usec"], lgdo.Array)
        assert isinstance(tbl["abs_delta_mu_usec"], lgdo.Array)
        assert isinstance(tbl["to_start_sec"], lgdo.Array)
        assert isinstance(tbl["to_start_usec"], lgdo.Array)
        assert isinstance(tbl["dr_start_pps"], lgdo.Array)
        assert isinstance(tbl["dr_start_ticks"], lgdo.Array)
        assert isinstance(tbl["dr_stop_pps"], lgdo.Array)
        assert isinstance(tbl["dr_stop_ticks"], lgdo.Array)
        assert isinstance(tbl["dr_maxticks"], lgdo.Array)
        assert isinstance(tbl["waveform"], lgdo.Struct)
        assert isinstance(tbl["waveform"]["t0"], lgdo.Array)
        assert isinstance(tbl["waveform"]["dt"], lgdo.Array)
        assert isinstance(tbl["waveform"]["values"], lgdo.ArrayOfEqualSizedArrays)


def test_values(event_rbkd, fcio_obj):
    fc = fcio_obj
    for ii, ch in enumerate(fc.event.trace_list):
        key = get_key(
            fc.config.streamid,
            fc.config.tracemap[ch] >> 16,
            fc.config.tracemap[ch] & 0xFFFF,
        )
        loc = event_rbkd[key].loc - 1
        tbl = event_rbkd[key].lgdo

        assert event_rbkd[key].fill_safety == 1

        assert tbl["packet_id"].nda[loc] == 69
        assert tbl["eventnumber"].nda[loc] == fc.event.timestamp[0]
        assert tbl["timestamp"].nda[loc] == fc.event.unix_time_utc_sec
        assert tbl["runtime"].nda[loc] == fc.event.run_time_sec[ii]
        assert tbl["lifetime"].nda[loc] == fc.event.life_time_sec[ii]
        assert tbl["deadtime"].nda[loc] == fc.event.dead_time_sec[ii]
        assert tbl["numtraces"].nda[loc] == fc.event.num_traces
        assert tbl["baseline"].nda[loc], fc.event.fpga_baseline[ii]
        assert tbl["daqenergy"].nda[loc], fc.event.fpga_energy[ii]
        assert tbl["channel"].nda[loc] == ch
        assert tbl["ts_pps"].nda[loc] == fc.event.timestamp[1]
        assert tbl["ts_ticks"].nda[loc] == fc.event.timestamp[2]
        assert tbl["ts_maxticks"].nda[loc] == fc.event.timestamp[3]
        assert tbl["mu_offset_sec"].nda[loc] == fc.event.timeoffset[0]
        assert tbl["mu_offset_usec"].nda[loc] == fc.event.timeoffset[1]
        assert tbl["to_master_sec"].nda[loc] == fc.event.timeoffset[2]
        assert tbl["delta_mu_usec"].nda[loc] == fc.event.timeoffset[3]
        assert tbl["abs_delta_mu_usec"].nda[loc] == fc.event.timeoffset[4]
        assert tbl["to_start_sec"].nda[loc] == fc.event.timeoffset[5]
        assert tbl["to_start_usec"].nda[loc] == fc.event.timeoffset[6]
        assert tbl["dr_start_pps"].nda[loc] == fc.event.deadregion[0]
        assert tbl["dr_start_ticks"].nda[loc] == fc.event.deadregion[1]
        assert tbl["dr_stop_pps"].nda[loc] == fc.event.deadregion[2]
        assert tbl["dr_stop_ticks"].nda[loc] == fc.event.deadregion[3]
        assert tbl["dr_maxticks"].nda[loc] == fc.event.deadregion[4]
        if fc.event.deadregion_size == 7:
            assert tbl["dr_ch_idx"].nda[loc] == fc.event.deadregion[5]
            assert tbl["dr_ch_len"].nda[loc] == fc.event.deadregion[6]
        else:
            assert tbl["dr_ch_idx"].nda[loc] == 0
            assert tbl["dr_ch_len"].nda[loc] == fc.config.adcs
        assert tbl["waveform"]["t0"].nda[loc] == 0
        assert tbl["waveform"]["dt"].nda[loc] == fc.config.sampling_period_ns
        assert np.array_equal(tbl["waveform"]["values"].nda[loc], fc.event.trace[ii])
