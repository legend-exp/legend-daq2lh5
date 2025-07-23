from __future__ import annotations

import copy
import logging

import lgdo
import numpy as np

from ..data_decoder import DataDecoder

log = logging.getLogger(__name__)


def get_bc(board: int, channel: int) -> int:
    """Create a standard hash for the board and channel of a CoMPASS file."""
    return (board << 9) + (channel & 0xF)


compass_decoded_values = {
    # packet index in file
    "packet_id": {"dtype": "uint32"},
    # ID of board
    "board": {"dtype": "uint32"},
    # ID of channel recording data
    "channel": {"dtype": "uint32"},
    # Timestamp of event
    "timestamp": {"dtype": "float64", "units": "ps"},
    # Energy of event in channels
    "energy": {"dtype": "uint32"},
    # Energy of event, calibrated
    "energy_calibrated": {"dtype": "float64"},
    # Energy short of event
    "energy_short": {"dtype": "uint32"},
    # Flags that the digitizer raised
    "flags": {"dtype": "uint32"},
    # number of samples in a waveform
    "num_samples": {"dtype": "int64"},
    # waveform data
    "waveform": {
        "dtype": "uint16",
        "datatype": "waveform",
        "wf_len": 65532,  # max value. override this before initializing buffers to save RAM
        "dt": 16,  # override if a different clock rate is used
        "dt_units": "ns",
        "t0_units": "ns",
    },
}
"""Default CoMPASS Event decoded values.

Warning
-------
This configuration can be dynamically modified by the decoder at runtime.
"""


class CompassEventDecoder(DataDecoder):
    """
    Decode CAEN digitizer event data.
    """

    def __init__(self, header=None, *args, **kwargs) -> None:
        self.decoded_values = {}
        super().__init__(*args, **kwargs)
        self.skipped_channels = {}
        if header is not None:
            self.set_header(header)

    # set the decoded_values for each board/channel combination because they could have different settings
    def set_header(self, header):
        self.header = header

        # Loop over crates, cards, build decoded values for enabled channels
        for board in self.header["boards"].keys():
            for channel in self.header["boards"][board]["channels"].keys():
                bc = get_bc(int(board), int(channel))

                self.decoded_values[bc] = copy.deepcopy(compass_decoded_values)

                # get trace length(s). Should all be the same
                self.decoded_values[bc]["waveform"]["wf_len"] = int(
                    float(
                        self.header["boards"][board]["wf_len"].value
                    )  # the header is a struct, so we need to return its value
                )

    def get_key_lists(self) -> list[list[str]]:
        key_lists = []
        for key in self.decoded_values.keys():
            key_lists.append(key)
        return [key_lists]

    def get_decoded_values(self, key=None):
        if key is None:
            dec_vals_list = self.decoded_values.values()
            if len(dec_vals_list) >= 0:
                return list(dec_vals_list)[0]
            raise RuntimeError("decoded_values not built")
        if key in self.decoded_values:
            return self.decoded_values[key]
        raise KeyError(f"no decoded values for key {key}")

    def decode_packet(
        self,
        packet: bytes,
        evt_rbkd: lgdo.Table | dict[int, lgdo.Table],
        packet_id: int,
        header: lgdo.Table | dict[int, lgdo.Table],
    ) -> bool:
        """Access ``CoMPASSEvent`` members for each event in the DAQ file.

        Parameters
        ----------
        packet
            The packet to be decoded
        evt_rbkd
            A single table for reading out all data, or a dictionary of tables
            keyed by channel number.
        packet_id
            The index of the packet in the `CoMPASS` stream. Incremented by
            :class:`~.compass.compass_streamer.CompassStreamer`.
        header
            The header of the CoMPASS file, along with user config info,
            that is used to determine waveform lengths and thus buffer sizes

        Returns
        -------
        n_bytes
            (estimated) number of bytes in the packet that was just decoded.
        """
        # Read in the board number and channel number so we can get the right entry in the table
        # Regardless of the length of the header, the first 4 bytes are the board and the channel number
        board = np.frombuffer(packet[0:2], dtype=np.uint16)[0]
        channel = np.frombuffer(packet[2:4], dtype=np.uint16)[0]

        bc = get_bc(board, channel)

        # get the table for this channel
        if bc not in evt_rbkd:
            if bc not in self.skipped_channels:
                self.skipped_channels[bc] = 0
                log.debug(f"Skipping channel: {channel}")
                log.debug(f"evt_rbkd: {evt_rbkd.keys()}")
            self.skipped_channels[bc] += 1
            return False
        tbl = evt_rbkd[bc].lgdo
        ii = evt_rbkd[bc].loc

        # store packet id
        tbl["packet_id"].nda[ii] = packet_id

        # store the info we already have read in
        tbl["board"].nda[ii] = board
        tbl["channel"].nda[ii] = channel

        # the time stamp also does not care about if we have an energy short present
        tbl["timestamp"].nda[ii] = np.frombuffer(packet[4:12], dtype=np.uint64)[0]

        # stumble our way through the energy, depending on what the header says
        bytes_read = 12
        if int(header["energy_channels"].value) == 1:
            tbl["energy"].nda[ii] = np.frombuffer(packet[12:14], dtype=np.uint16)[0]
            bytes_read += 2
            if int(header["energy_calibrated"].value) == 1:
                tbl["energy_calibrated"].nda[ii] = None
        elif (int(header["energy_calibrated"].value) == 1) and (
            int(header["energy_channels"].value) == 0
        ):
            tbl["energy_calibrated"].nda[ii] = np.frombuffer(
                packet[14:22], dtype=np.float64
            )[0]
            bytes_read += 8
            tbl["energy"].nda[ii] = None
        else:
            tbl["energy_calibrated"].nda[ii] = np.frombuffer(
                packet[12:20], dtype=np.float64
            )[0]
            bytes_read += 8

        # now handle the energy short
        if int(header["energy_short"].value) == 1:
            tbl["energy_short"].nda[ii] = np.frombuffer(
                packet[bytes_read : bytes_read + 2], dtype=np.uint16
            )[0]
            bytes_read += 2
        else:
            tbl["energy_short"].nda[ii] = None

        tbl["flags"].nda[ii] = np.frombuffer(
            packet[bytes_read : bytes_read + 4], np.uint32
        )[0]
        bytes_read += 5  # skip over the waveform code
        tbl["num_samples"].nda[ii] = np.frombuffer(
            packet[bytes_read : bytes_read + 4], dtype=np.uint32
        )[0]
        bytes_read += 4

        if (
            tbl["num_samples"].nda[ii] != self.decoded_values[bc]["waveform"]["wf_len"]
        ):  # make sure that the waveform we read in is the same length as in the config
            raise RuntimeError(
                f"Waveform size {tbl['num_samples'].nda[ii]} doesn't match expected size {self.decoded_values[bc]['waveform']['wf_len']}. "
                "Skipping packet"
            )

        tbl["waveform"]["values"].nda[ii] = np.frombuffer(
            packet[bytes_read:], dtype=np.uint16
        )

        evt_rbkd[bc].loc += 1
        return evt_rbkd[bc].is_full()
