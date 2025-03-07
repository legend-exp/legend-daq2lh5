import json
import os
from pathlib import Path

import h5py
import pytest
from lgdo import lh5
from lgdo.compression import ULEB128ZigZagDiff

from daq2lh5 import build_raw
from daq2lh5.fc.fc_event_decoder import fc_event_decoded_values

config_dir = Path(__file__).parent / "configs"


def test_build_raw_basics(lgnd_test_data):
    with pytest.raises(FileNotFoundError):
        build_raw(in_stream="non-existent-file")

    with pytest.raises(FileNotFoundError):
        build_raw(
            in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
            out_spec="non-existent-file.json",
        )


def test_build_raw_fc(lgnd_test_data, tmptestdir):
    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        overwrite=True,
    )

    out_file = lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.lh5")
    assert lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.lh5") != ""

    with pytest.raises(FileExistsError):
        build_raw(
            in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio")
        )
    os.remove(out_file)

    out_file = f"{tmptestdir}/L200-comm-20211130-phy-spms.lh5"

    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        out_spec=out_file,
        overwrite=True,
    )

    assert os.path.exists(out_file)


def test_build_raw_fc_ghissue10(lgnd_test_data, tmptestdir):
    out_file = f"{tmptestdir}/l200-p06-r007-cal-20230725T202227Z.lh5"
    build_raw(
        in_stream=lgnd_test_data.get_path(
            "fcio/l200-p06-r007-cal-20230725T202227Z.fcio"
        ),
        out_spec=out_file,
        buffer_size=123,
        overwrite=True,
    )

    assert os.path.exists(out_file)


def test_invalid_user_buffer_size(lgnd_test_data, tmptestdir):
    with pytest.raises(ValueError):
        build_raw(
            in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
            buffer_size=5,
            overwrite=True,
        )


def test_build_raw_fc_out_spec(lgnd_test_data, tmptestdir):
    out_file = f"{tmptestdir}/L200-comm-20211130-phy-spms.lh5"
    out_spec = {
        "FCEventDecoder": {
            "spms": {"key_list": [[52802, 52804]], "out_stream": out_file}
        }
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        out_spec=out_spec,
        n_max=10 * 3,  # decode 10 events of 3 channels per record into one table
        overwrite=True,
    )

    store = lh5.LH5Store()
    lh5_obj = store.read("/spms", out_file)
    assert len(lh5_obj) == 10
    assert (lh5_obj["channel"].nda == [2, 3, 4, 2, 3, 4, 2, 3, 4, 2]).all()

    with open(f"{config_dir}/fc-out-spec.json") as f:
        out_spec = json.load(f)

    out_spec["FCEventDecoder"]["spms"]["out_stream"] = out_spec["FCEventDecoder"][
        "spms"
    ]["out_stream"].replace("/tmp", f"{tmptestdir}")

    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        out_spec=out_spec,
        n_max=10,
        overwrite=True,
    )


def test_build_raw_fc_channelwise_out_spec(lgnd_test_data, tmptestdir):
    out_file = f"{tmptestdir}/L200-comm-20211130-phy-spms.lh5"
    out_spec = {
        "FCEventDecoder": {
            "ch{key}": {
                "key_list": [[52800, 52806]],
                "out_stream": out_file + ":{name}",
                "out_name": "raw",
            }
        }
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
        out_spec=out_spec,
        overwrite=True,
    )

    assert lh5.ls(out_file) == [
        "ch52800",
        "ch52801",
        "ch52802",
        "ch52803",
        "ch52804",
        "ch52805",
    ]
    assert lh5.ls(out_file, "ch52800/") == ["ch52800/raw"]
    assert lh5.ls(out_file, "ch52800/raw/waveform") == ["ch52800/raw/waveform"]


def test_build_raw_orca(lgnd_test_data, tmptestdir):
    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        overwrite=True,
    )

    assert lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.lh5") != ""

    out_file = f"{tmptestdir}/L200-comm-20220519-phy-geds.lh5"

    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        out_spec=out_file,
        overwrite=True,
    )

    assert os.path.exists(f"{tmptestdir}/L200-comm-20220519-phy-geds.lh5")


def test_build_raw_orca_out_spec(lgnd_test_data, tmptestdir):
    out_file = f"{tmptestdir}/L200-comm-20220519-phy-geds.lh5"
    out_spec = {
        "ORFlashCamADCWaveformDecoder": {
            "geds": {"key_list": [[1028802, 1028804]], "out_stream": out_file}
        }
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        out_spec=out_spec,
        n_max=10,
        overwrite=True,
    )

    store = lh5.LH5Store()
    lh5_obj = store.read("/geds", out_file)
    assert len(lh5_obj) == 10
    assert (lh5_obj["channel"].nda == [2, 3, 4, 2, 3, 4, 2, 3, 4, 2]).all()

    with open(f"{config_dir}/orca-out-spec.json") as f:
        out_spec = json.load(f)

    out_spec["ORFlashCamADCWaveformDecoder"]["geds"]["out_stream"] = out_spec[
        "ORFlashCamADCWaveformDecoder"
    ]["geds"]["out_stream"].replace("/tmp", f"{tmptestdir}")

    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        out_spec=out_spec,
        n_max=10,
        overwrite=True,
    )


def test_build_raw_hdf5_settings(lgnd_test_data, tmptestdir):
    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        hdf5_settings={"compression": "lzf", "shuffle": False},
        overwrite=True,
    )

    with h5py.File(
        lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.lh5")
    ) as f:
        assert f["ORFlashCamADCWaveform/abs_delta_mu_usec"].shuffle is False
        assert f["ORFlashCamADCWaveform/abs_delta_mu_usec"].compression == "lzf"


def test_build_raw_hdf5_settings_in_decoded_values(lgnd_test_data, tmptestdir):
    fc_event_decoded_values["packet_id"]["hdf5_settings"] = {
        "shuffle": False,
        "compression": "lzf",
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        overwrite=True,
    )

    del fc_event_decoded_values["packet_id"]["hdf5_settings"]

    with h5py.File(
        lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.lh5")
    ) as f:
        assert f["ORFlashCamADCWaveform/packet_id"].shuffle is False
        assert f["ORFlashCamADCWaveform/packet_id"].compression == "lzf"


def test_build_raw_wf_compression_in_decoded_values(lgnd_test_data, tmptestdir):
    out_file = lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.lh5")

    fc_event_decoded_values["waveform"].setdefault(
        "hdf5_settings", {"values": {}, "t0": {}}
    )
    fc_event_decoded_values["waveform"]["hdf5_settings"] = {
        "values": {"shuffle": False, "compression": "lzf"},
        "t0": {"shuffle": True, "compression": None},
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        overwrite=True,
    )

    with h5py.File(out_file) as f:
        assert f["ORFlashCamADCWaveform/waveform/values"].shuffle is False
        assert f["ORFlashCamADCWaveform/waveform/values"].compression == "lzf"
        assert f["ORFlashCamADCWaveform/waveform/t0"].shuffle is True
        assert f["ORFlashCamADCWaveform/waveform/t0"].compression is None

    fc_event_decoded_values["waveform"].setdefault("compression", {"values": None})
    fc_event_decoded_values["waveform"]["compression"]["values"] = ULEB128ZigZagDiff()

    build_raw(
        in_stream=lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        overwrite=True,
    )

    del fc_event_decoded_values["waveform"]["hdf5_settings"]
    del fc_event_decoded_values["waveform"]["compression"]

    with h5py.File(out_file) as f:
        assert (
            f[
                "ORFlashCamADCWaveform/waveform/values/encoded_data/flattened_data"
            ].compression
            is None
        )
        assert f["ORFlashCamADCWaveform/waveform/t0"].shuffle is True
        assert f["ORFlashCamADCWaveform/waveform/t0"].compression is None

    store = lh5.LH5Store()
    obj = store.read(
        "ORFlashCamADCWaveform/waveform/values", out_file, decompress=False
    )
    assert obj.attrs["codec"] == "uleb128_zigzag_diff"


def test_build_raw_compass(lgnd_test_data, tmptestdir):
    build_raw(
        in_stream=lgnd_test_data.get_path("compass/compass_test_data.BIN"),
        overwrite=True,
        compass_config_file=lgnd_test_data.get_path(
            "compass/compass_test_data_settings.xml"
        ),
    )

    assert lgnd_test_data.get_path("compass/compass_test_data.lh5") != ""

    out_file = f"{tmptestdir}/compass_test_data.lh5"

    build_raw(
        in_stream=lgnd_test_data.get_path("compass/compass_test_data.BIN"),
        out_spec=out_file,
        overwrite=True,
        compass_config_file=lgnd_test_data.get_path(
            "compass/compass_test_data_settings.xml"
        ),
    )

    assert os.path.exists(f"{tmptestdir}/compass_test_data.lh5")


def test_build_raw_compass_out_spec(lgnd_test_data, tmptestdir):
    out_file = f"{tmptestdir}/compass_test_data.lh5"
    out_spec = {
        "CompassEventDecoder": {"spms": {"key_list": [[0, 1]], "out_stream": out_file}}
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("compass/compass_test_data.BIN"),
        out_spec=out_spec,
        n_max=10,
        overwrite=True,
        compass_config_file=lgnd_test_data.get_path(
            "compass/compass_test_data_settings.xml"
        ),
    )

    store = lh5.LH5Store()
    lh5_obj = store.read("/spms", out_file)
    assert len(lh5_obj) == 10
    assert (lh5_obj["channel"].nda == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).all()


def test_build_raw_compass_out_spec_no_config(lgnd_test_data, tmptestdir):
    out_file = f"{tmptestdir}/compass_test_data.lh5"
    out_spec = {
        "CompassEventDecoder": {"spms": {"key_list": [[0, 1]], "out_stream": out_file}}
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("compass/compass_test_data.BIN"),
        out_spec=out_spec,
        n_max=10,
        overwrite=True,
    )

    store = lh5.LH5Store()
    lh5_obj = store.read("/spms", out_file)
    assert len(lh5_obj) == 10
    assert (lh5_obj["channel"].nda == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).all()


def test_build_raw_orca_sis3316(lgnd_test_data, tmptestdir):
    out_file = f"{tmptestdir}/coherent-run1141-bkg.lh5"
    out_spec = {
        "ORSIS3316WaveformDecoder": {
            "Card1": {"key_list": [48], "out_stream": out_file}
        }
    }

    build_raw(
        in_stream=lgnd_test_data.get_path("orca/sis3316/coherent-run1141-bkg.orca"),
        out_spec=out_spec,
        n_max=10,
        overwrite=True,
    )

    assert os.path.exists(out_file)
