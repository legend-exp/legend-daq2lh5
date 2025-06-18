import pytest

from daq2lh5.orca import orca_packet


@pytest.fixture(scope="module")
def fcio_packets(orca_stream_fcio):
    packets = []
    packets.append(orca_stream_fcio.load_packet(3).copy())  # config
    packets.append(orca_stream_fcio.load_packet(6).copy())  # event
    orca_stream_fcio.close_stream()  # avoid warning that file is still open
    return packets


def test_orfcio_config_decoding(orca_stream_fcio, fcio_packets):
    config_packet = fcio_packets[0]
    assert config_packet is not None

    data_id = orca_packet.get_data_id(config_packet)
    name = orca_stream_fcio.header.get_id_to_decoder_name_dict()[data_id]
    assert name == "ORFCIOConfigDecoder"


def test_orfcio_waveform_decoding(orca_stream_fcio, fcio_packets):
    wf_packet = fcio_packets[1]
    assert wf_packet is not None

    data_id = orca_packet.get_data_id(wf_packet)
    name = orca_stream_fcio.header.get_id_to_decoder_name_dict()[data_id]
    assert name == "ORFCIOEventDecoder"


@pytest.fixture(scope="module")
def fcio_swt_packets(orca_stream_fcio_swt):
    packets = []
    packets.append(orca_stream_fcio_swt.load_packet(3).copy())  # config
    packets.append(orca_stream_fcio_swt.load_packet(6).copy())  # event
    orca_stream_fcio_swt.close_stream()  # avoid warning that file is still open
    return packets


def test_orfcio_config_decoding_swt(orca_stream_fcio_swt, fcio_swt_packets):
    config_packet = fcio_swt_packets[0]
    assert config_packet is not None

    data_id = orca_packet.get_data_id(config_packet)
    name = orca_stream_fcio_swt.header.get_id_to_decoder_name_dict()[data_id]
    assert name == "ORFCIOConfigDecoder"


def test_orfcio_waveform_decoding_swt(orca_stream_fcio_swt, fcio_swt_packets):
    wf_packet = fcio_swt_packets[1]
    assert wf_packet is not None

    data_id = orca_packet.get_data_id(wf_packet)
    name = orca_stream_fcio_swt.header.get_id_to_decoder_name_dict()[data_id]
    assert name == "ORFCIOEventHeaderDecoder"
