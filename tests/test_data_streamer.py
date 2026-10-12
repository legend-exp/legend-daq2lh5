import lh5

from daq2lh5 import build_raw
from daq2lh5.fc.fc_streamer import FCStreamer
from daq2lh5.raw_buffer import RawBuffer


def test_iteration_matches_build_raw(lgnd_test_data, tmptestdir):
    in_file = lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio")

    rows = {}
    n_chunks = 0
    streamer = FCStreamer()
    streamer.open_stream(in_file, buffer_size=6)
    for chunk_list in streamer:
        n_chunks += 1
        assert isinstance(chunk_list, list)
        for rb in chunk_list:
            assert isinstance(rb, RawBuffer)
            assert rb.loc > 0
            rows[rb.out_name] = rows.get(rb.out_name, 0) + rb.loc
    streamer.close_stream()

    assert n_chunks > 1  # buffer_size=6 forces multiple chunks
    assert rows  # we decoded something

    # the disk-writing path must agree on the number of decoded rows
    out_file = f"{tmptestdir}/test_data_streamer_iter.lh5"
    build_raw(in_file, out_spec=out_file, buffer_size=6, overwrite=True)
    for out_name, n_rows in rows.items():
        assert n_rows == lh5.read_n_rows(out_name, out_file)


def test_iteration_clears_buffers_and_terminates(lgnd_test_data):
    in_file = lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio")

    streamer = FCStreamer()
    streamer.open_stream(in_file, buffer_size=1024)
    chunks = list(streamer)
    assert len(chunks) > 0
    # buffers were cleared after the final yield
    for rb_list in streamer.rb_lib.values():
        for rb in rb_list:
            assert rb.loc == 0
    # stream is exhausted: a fresh iteration yields nothing
    assert list(streamer) == []
    streamer.close_stream()
