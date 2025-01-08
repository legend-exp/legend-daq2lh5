from daq2lh5.llama.llama_event_decoder import LLAMAEventDecoder
from daq2lh5.llama.llama_header_decoder import LLAMAHeaderDecoder
from daq2lh5.llama.llama_streamer import LLAMAStreamer
from daq2lh5.raw_buffer import RawBuffer


def test_get_decoder_list():
    streamer = LLAMAStreamer()
    assert len(streamer.get_decoder_list()) == 2
    assert isinstance(streamer.get_decoder_list()[0], LLAMAHeaderDecoder)
    assert isinstance(streamer.get_decoder_list()[1], LLAMAEventDecoder)


# test_data_path (str) from fixture in ./conftest.py
def test_open_stream(test_data_path):
    streamer = LLAMAStreamer()
    rbl: list[RawBuffer] = streamer.open_stream(test_data_path)
    assert len(rbl) == 1
    assert isinstance(rbl[0], RawBuffer)
    assert streamer.rb_lib is not None
    nbytes_hdr = streamer.n_bytes_read
    assert nbytes_hdr > 0
    assert streamer.read_packet()  # there has to be at last a single good packet
    assert streamer.packet_id == 1
    assert streamer.n_bytes_read > nbytes_hdr
    streamer.close_stream()


def test_open_stream_multiple(test_data_path):
    streamer = LLAMAStreamer()
    rbl: list[RawBuffer] = streamer.open_stream(test_data_path)
    assert len(rbl) == 1
    streamer.close_stream()
    rbl: list[RawBuffer] = streamer.open_stream(test_data_path)
    assert len(rbl) == 1
    streamer.close_stream()
