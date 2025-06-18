from daq2lh5.orca import orca_packet


def test_orca_packet_funcs(orca_stream):
    # The values in this test are particular to the test orca file in
    # legend-testdata and may need to be changed if that file is changed

    assert orca_stream.count_packets() == 911

    packet = orca_stream.load_packet()
    assert orca_packet.is_short(packet) is False
    assert orca_packet.get_data_id(packet) == 3
    assert orca_packet.get_n_words(packet) == 4
    assert orca_packet.hex_dump(packet, return_output=True)[-1] == "3 0x63c1977a"

    id_dict = orca_stream.header.get_id_to_decoder_name_dict()
    seen = []
    for ii in range(100):
        packet = orca_stream.load_packet(ii)
        if packet is None:
            break
        name = id_dict[orca_packet.get_data_id(packet)]
        # if ii < 20: print(ii, name)
        if ii == 0:
            assert name == "OrcaHeaderDecoder"
        if ii == 1:
            assert name == "ORRunDecoderForRun"
        if ii == 910:
            assert name == "ORRunDecoderForRun"
        if name not in seen:
            seen.append(name)
    expected = [
        "OrcaHeaderDecoder",
        "ORRunDecoderForRun",
        "ORFlashCamListenerConfigDecoder",
        "ORFlashCamListenerStatusDecoder",
        "ORFlashCamWaveformDecoder",
    ]
    assert seen == expected

    orca_stream.close_stream()  # avoid warning that file is still open


def test_orca_packet_funcs_fcio(orca_stream_fcio):
    # The values in this test are particular to the test orca file in
    # legend-testdata and may need to be changed if that file is changed

    assert orca_stream_fcio.count_packets() == 13

    packet = orca_stream_fcio.load_packet()
    assert orca_packet.is_short(packet) is False
    assert orca_packet.get_data_id(packet) == 3
    assert orca_packet.get_n_words(packet) == 4
    assert orca_packet.hex_dump(packet, return_output=True)[-1] == "3 0x68423e20"

    id_dict = orca_stream_fcio.header.get_id_to_decoder_name_dict()
    seen = []
    for ii in range(100):
        packet = orca_stream_fcio.load_packet(ii)
        if packet is None:
            break
        name = id_dict[orca_packet.get_data_id(packet)]
        if ii == 0:
            assert name == "OrcaHeaderDecoder"
        if ii == 1:
            assert name == "ORRunDecoderForRun"
        if ii == 910:
            assert name == "ORRunDecoderForRun"
        if name not in seen:
            seen.append(name)
    expected = [
        "OrcaHeaderDecoder",
        "ORRunDecoderForRun",
        "ORFCIOConfigDecoder",
        "ORFCIOEventDecoder",
    ]
    assert seen == expected

    orca_stream_fcio.close_stream()  # avoid warning that file is still open


def test_orca_packet_funcs_fcio_swt(orca_stream_fcio_swt):
    # The values in this test are particular to the test orca file in
    # legend-testdata and may need to be changed if that file is changed

    assert orca_stream_fcio_swt.count_packets() == 13

    packet = orca_stream_fcio_swt.load_packet()
    assert orca_packet.is_short(packet) is False
    assert orca_packet.get_data_id(packet) == 3
    assert orca_packet.get_n_words(packet) == 4
    assert orca_packet.hex_dump(packet, return_output=True)[-1] == "3 0x67748ec3"

    id_dict = orca_stream_fcio_swt.header.get_id_to_decoder_name_dict()
    seen = []
    for ii in range(100):
        packet = orca_stream_fcio_swt.load_packet(ii)
        if packet is None:
            break
        name = id_dict[orca_packet.get_data_id(packet)]
        if ii == 0:
            assert name == "OrcaHeaderDecoder"
        if ii == 1:
            assert name == "ORRunDecoderForRun"
        if ii == 910:
            assert name == "ORRunDecoderForRun"
        if name not in seen:
            seen.append(name)
    expected = [
        "OrcaHeaderDecoder",
        "ORRunDecoderForRun",
        "ORFCIOConfigDecoder",
        "ORFCIOEventHeaderDecoder",
    ]
    assert seen == expected

    orca_stream_fcio_swt.close_stream()  # avoid warning that file is still open
