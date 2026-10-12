from pathlib import Path

import pytest

from daq2lh5 import build_raw
from daq2lh5.orca.orca_streamer import OrcaStreamer


def _load_all(path):
    orstr = OrcaStreamer()
    orstr.open_stream(str(path))
    packets = []
    while (pkt := orstr.load_packet()) is not None:
        packets.append(pkt.tobytes())
    locs = list(orstr.packet_locs)
    n_resyncs = orstr.n_resyncs
    orstr.close_stream()
    return packets, locs, n_resyncs


@pytest.fixture(scope="module")
def corrupted_orca(lgnd_test_data, tmptestdir):
    """Test file with a packet cut short by a 3-byte (non word aligned) blob."""
    path = lgnd_test_data.get_path("orca/fc/l200-p02-r008-phy-20230113T174010Z.orca")
    clean, locs, _ = _load_all(path)
    raw = Path(path).read_bytes()
    k = 20  # packet index (0 is the header)
    cut = locs[k] + (locs[k + 1] - locs[k]) // 2
    out = Path(tmptestdir) / "corrupted.orca"
    out.write_bytes(raw[:cut] + b"\xaa\xbb\xcc" + raw[locs[k + 1] :])
    return out, clean, k


def test_resync_skips_corruption(corrupted_orca):
    path, clean, k = corrupted_orca
    packets, _, n_resyncs = _load_all(path)
    assert n_resyncs == 1
    # only the damaged packet and the one it swallows may be lost
    assert packets[: k - 1] == clean[: k - 1]
    tail = clean[k + 2 :]
    assert packets[-len(tail) :] == tail
    assert len(clean) - 2 <= len(packets) <= len(clean)


def test_resync_build_raw(corrupted_orca, tmptestdir):
    path, _, _ = corrupted_orca
    build_raw(str(path), out_spec=f"{tmptestdir}/corrupted.lh5", overwrite=True)


def test_clean_file_no_resync(lgnd_test_data):
    path = lgnd_test_data.get_path("orca/fc/l200-p02-r008-phy-20230113T174010Z.orca")
    assert _load_all(path)[2] == 0
