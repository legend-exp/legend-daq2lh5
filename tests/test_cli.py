import os
import subprocess
from pathlib import Path

import h5py

config_dir = Path(__file__).parent / "configs"


def test_build_raw_cli(lgnd_test_data, tmptestdir):
    subprocess.check_call(["legend-daq2lh5", "--help"])
    subprocess.check_call(
        [
            "legend-daq2lh5",
            "--overwrite",
            "--stream-type",
            "ORCA",
            "--out-spec",
            f"{config_dir}/orca-out-spec-cli.json",
            "--max-rows",
            "10",
            "--buffer_size",
            "8192",
            lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
        ]
    )

    assert os.path.exists("/tmp/L200-comm-20220519-phy-geds.lh5")
    os.remove("/tmp/L200-comm-20220519-phy-geds.lh5")


def test_build_raw_cli_kwargs(lgnd_test_data, tmptestdir):
    subprocess.check_call(["legend-daq2lh5", "--help"])
    kwarg_string = '{"hdf5_settings":{"compression":"lzf"}}'
    subprocess.check_call(
        [
            "legend-daq2lh5",
            "--overwrite",
            "--stream-type",
            "ORCA",
            "--out-spec",
            f"{config_dir}/orca-out-spec-cli.json",
            "--max-rows",
            "10",
            "--buffer_size",
            "8192",
            lgnd_test_data.get_path("orca/fc/L200-comm-20220519-phy-geds.orca"),
            "--kwargs",
            f"{kwarg_string}",
        ]
    )

    assert os.path.exists("/tmp/L200-comm-20220519-phy-geds.lh5")
    with h5py.File("/tmp/L200-comm-20220519-phy-geds.lh5", "r") as f:
        assert f["g1028800/baseline"].compression == "lzf"
    os.remove("/tmp/L200-comm-20220519-phy-geds.lh5")
