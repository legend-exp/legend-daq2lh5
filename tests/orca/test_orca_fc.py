import lh5
import numpy as np
import pytest

from daq2lh5 import build_raw
from daq2lh5.orca import orca_packet


@pytest.fixture(scope="module")
def fc_packets(orca_stream):
    packets = []
    packets.append(orca_stream.load_packet(3).copy())  # config
    packets.append(orca_stream.load_packet(4).copy())  # status
    packets.append(orca_stream.load_packet(13).copy())  # waveform
    orca_stream.close_stream()  # avoid warning that file is still open
    return packets


def test_orfc_config_decoding(orca_stream, fc_packets):
    config_packet = fc_packets[0]
    assert config_packet is not None

    data_id = orca_packet.get_data_id(config_packet)
    name = orca_stream.header.get_id_to_decoder_name_dict()[data_id]
    assert name == "ORFlashCamListenerConfigDecoder"


def test_orfc_status_decoding(orca_stream, fc_packets):
    status_packet = fc_packets[1]
    assert status_packet is not None

    data_id = orca_packet.get_data_id(status_packet)
    name = orca_stream.header.get_id_to_decoder_name_dict()[data_id]
    assert name == "ORFlashCamListenerStatusDecoder"


def test_orfc_waveform_decoding(orca_stream, fc_packets):
    wf_packet = fc_packets[2]
    assert wf_packet is not None

    data_id = orca_packet.get_data_id(wf_packet)
    name = orca_stream.header.get_id_to_decoder_name_dict()[data_id]
    assert name == "ORFlashCamWaveformDecoder"


def test_orfc_waveform_dead_defaults(lgnd_test_data, tmptestdir):
    """Fields absent from ORCA packets get deterministic defaults."""
    out = f"{tmptestdir}/orca-fc-dead-defaults.lh5"
    build_raw(
        lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        out_spec=out,
        overwrite=True,
    )
    tbl = lh5.read("ORFlashCamADCWaveform", out)
    n_adc = len(np.unique(tbl["channel"].nda))
    assert (tbl["dr_ch_idx"].nda == 0).all()
    assert (tbl["dr_ch_len"].nda == n_adc).all()
    assert (tbl["deadinterval_nsec"].nda == 0).all()
    assert (tbl["lifetime"].nda == 0).all()
