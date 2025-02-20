from __future__ import annotations

import copy
import logging
from typing import Any

from daq2lh5.raw_buffer import RawBuffer, RawBufferList
from fcio import FCIO, Tags, Limits
import lgdo

from lgdo.types import vovutils
import numpy as np

from ..data_decoder import DataDecoder

import line_profiler

log = logging.getLogger(__name__)

def get_key(streamid: int, card_address: int, card_input: int, iwf: int = -1) -> int:
    if streamid > 0 or iwf < 0:
        # For backwards compatibility only the lower 16-bit of the streamid are
        # used.
        return (streamid & 0xFFFF) * 1000000 + card_address * 100 + card_input
    else:
        return iwf


def get_fcid(key: int) -> int:
    return int(key // 1000000)


def get_card_address(key: int) -> int:
    return int((key // 100) % 10000)


def get_card_input(key: int) -> int:
    return int(key % 100)


# put decoded values here where they can be used also by the orca decoder
fc_eventheader_decoded_values = {
    "packet_id": {
        "dtype": "uint32",
        "description": "The index of this decoded packet in the file.",
    },
    "fcid": {"dtype": "uint16", "description": "The ID of this data stream."},
    "board_id": {
        "dtype": "uint16",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "The MAC address of the FlashCam board which recorded this event.",
    },
    "fc_input": {
        "dtype": "uint16",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "The per-board ADC channel. Corresponds to the input connector in 62.5MHz mode or the input wire in 250MHz mode.",
    },
    "eventnumber": {
        "dtype": "int32",
        "description": "Counts triggered events. Fetched from the master board in non-sparse, or from the ADC board in sparse.",
    },
    "event_type": {
        "dtype": "int32",
        "description": "Is 1 for shared trigger/readout, or 11 in true sparse mode where ADC boards trigger independently.",
    },
    "timestamp": {
        "dtype": "float64",
        "units": "s",
        "description": "Time since epoch (unix time) for this event.",
    },
    "runtime": {
        "dtype": "float64",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "units": "s",
        "description": "Time since trigger enable of this run. Recorded per channel.",
    },
    "lifetime": {
        "dtype": "float64",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "units": "s",
        "description": "The accumulated lifetime since trigger enable, for which the hardware was not dead. Recorded per channel.",
    },
    "deadtime": {
        "dtype": "float64",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "units": "s",
        "description": "The accumulated time for which the hardware buffers were full (dead). Recorded per channel.",
    },
    "deadtime_nsec": {
        "dtype": "int64",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "units": "ns",
        "description": "Same as the deadtime field, but recorded in nanoseconds with 4ns precision (due to the internal clock speed of 250MHz).",
    },
    "deadinterval_nsec": {
        "dtype": "int64",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "units": "ns",
        "description": "The interval the board was dead between previous event and this event in nanoseconds with 4ns precision (due to the internal clock speed of 250MHz).",
    },
    "numtraces": {"dtype": "int32"},
    "tracelist": {
        "dtype": "uint16",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "The number of read out adc channels in this event.",
    },
    "baseline": {
        "dtype": "uint16",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "The baseline determined by the FPGA baseline algorithm.",
    },
    "daqenergy": {
        "dtype": "uint16",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": Limits.MaxChannels,
        "description": "The shaped pulse peak height above the trigger threshold for 62.5MHz mode. The waveform integral over `sumlength` samples in 250MHz mode.",
    },
    "ts_pps": {
        "dtype": "int32",
        "description": "The pulse-per-second (PPS) counter since DAQ reset.",
    },
    "ts_ticks": {
        "dtype": "int32",
        "description": "The clock ticks counter (250MHz) since last PPS.",
    },
    "ts_maxticks": {
        "dtype": "int32",
        "description": "The clock ticks counter value when the last PPS arrived.",
    },
    "mu_offset_sec": {
        "dtype": "int32",
        "description": "The seconds between server (unix) and fpga time. See `man 2 gettimeofday`.",
    },
    "mu_offset_usec": {
        "dtype": "int32",
        "description": "The microseconds between server (unix second) and fpga time (pps). See `man 2 gettimeofday`",
    },
    "to_master_sec": {
        "dtype": "int32",
        "description": "The timeoffset between server (unix) and fpga clocks, only valid when a gps clock/time server is used and the offset to epoch needs to be adjusted.",
    },
    "delta_mu_usec": {
        "dtype": "int32",
        "description": "The clock offset in microseconds between fpga and server for aligned clocks.",
    },
    "abs_delta_mu_usec": {
        "dtype": "int32",
        "description": "The absolute value of `delta_mu_usec`.",
    },
    "to_start_sec": {
        "dtype": "int32",
        "description": "The seconds of the start time of the run / trigger enable.",
    },
    "to_start_usec": {
        "dtype": "int32",
        "description": "The microseconds of the start time of the run / trigger enable.",
    },
    "dr_start_pps": {
        "dtype": "int32",
        "description": "The PPS counter value when the hardware buffers were full (dead), before this event. Only changes if a new value is aquired.",
    },
    "dr_start_ticks": {
        "dtype": "int32",
        "description": "The ticks counter value when the hardware buffers were full (dead), before this event. Only changes if a new value is aquired.",
    },
    "dr_stop_pps": {
        "dtype": "int32",
        "description": "The PPS counter value when the hardware buffers free (life) again, before this event. Only changes if a new value is aquired.",
    },
    "dr_stop_ticks": {
        "dtype": "int32",
        "description": "The ticks counter value when the hardware buffers free (life) again, before this event. Only changes if a new value is aquired.",
    },
    "dr_maxticks": {
        "dtype": "int32",
        "description": "The ticks counter value on PPS when the hardware buffers were freed, before this event. Only changes if a new value is aquired.",
    },
    "dr_ch_idx": {
        "dtype": "uint16",
        "description": "The start channel index of channels which are affected by the recorded dead interval.",
    },
    "dr_ch_len": {
        "dtype": "uint16",
        "description": "The number of channels which are affected by the dead interval, starts from `dr_ch_idx`. Can be larger than `num_traces`.",
    },
}


class FCEventHeaderDecoder(DataDecoder):
    def __init__(self, *args, **kwargs) -> None:
        self.decoded_values = copy.deepcopy(fc_eventheader_decoded_values)
        super().__init__(*args, **kwargs)
        self.max_rows_in_packet = (
            1  # only == 1 if len(array) in table is counted as one.
        )
        self.key_list = []

    def set_fcio_stream(self, fcio_stream: FCIO) -> None:
        """Access ``FCIOConfig`` members once when each file is opened.

        Parameters
        ----------
        fc_config
            extracted via :meth:`~.fc_config_decoder.FCConfigDecoder.decode_config`.
        """
        n_traces = len(fcio_stream.config.tracemap)
        # self.max_rows_in_packet = max(n_traces, self.max_rows_in_packet)

        self.decoded_values["tracelist"]["length_guess"] = n_traces
        self.decoded_values["board_id"]["length_guess"] = n_traces
        self.decoded_values["fc_input"]["length_guess"] = n_traces
        self.decoded_values["deadtime_nsec"]["length_guess"] = n_traces
        self.decoded_values["deadinterval_nsec"]["length_guess"] = n_traces
        self.decoded_values["deadtime"]["length_guess"] = n_traces
        self.decoded_values["runtime"]["length_guess"] = n_traces
        self.decoded_values["lifetime"]["length_guess"] = n_traces
        self.decoded_values["baseline"]["length_guess"] = n_traces
        self.decoded_values["daqenergy"]["length_guess"] = n_traces

        self.key_list = [get_key(fcio_stream.config.streamid, 0, 0)]

    def get_key_lists(self) -> list[list[int | str]]:
        return [copy.deepcopy(self.key_list)]

    def get_decoded_values(self, key: int | str = None) -> dict[str, dict[str, Any]]:
        # FC uses the same values for all channels
        return self.decoded_values

    def get_max_rows_in_packet(self) -> int:
        return self.max_rows_in_packet

    def decode_packet(
        self,
        fcio: FCIO,
        evt_hdr_rbkd: lgdo.Table | dict[int, lgdo.Table],
        packet_id: int,
    ) -> bool:

        # only one key available: this streamid
        evt_hdr_rb = evt_hdr_rbkd[self.key_list[0]]

        tbl = evt_hdr_rb.lgdo
        loc = evt_hdr_rb.loc

        tbl["packet_id"].nda[loc] = packet_id
        tbl["fcid"].nda[loc] = fcio.config.streamid & 0xFFFF
        tbl["event_type"].nda[loc] = fcio.event.type
        tbl["eventnumber"].nda[loc] = fcio.event.timestamp[0]
        tbl["ts_pps"].nda[loc] = fcio.event.timestamp[1]
        tbl["ts_ticks"].nda[loc] = fcio.event.timestamp[2]
        tbl["ts_maxticks"].nda[loc] = fcio.event.timestamp[3]
        tbl["mu_offset_sec"].nda[loc] = fcio.event.timeoffset[0]
        tbl["mu_offset_usec"].nda[loc] = fcio.event.timeoffset[1]
        tbl["to_master_sec"].nda[loc] = fcio.event.timeoffset[2]
        tbl["delta_mu_usec"].nda[loc] = fcio.event.timeoffset[3]
        tbl["abs_delta_mu_usec"].nda[loc] = fcio.event.timeoffset[4]
        tbl["to_start_sec"].nda[loc] = fcio.event.timeoffset[5]
        tbl["to_start_usec"].nda[loc] = fcio.event.timeoffset[6]
        tbl["dr_start_pps"].nda[loc] = fcio.event.deadregion[0]
        tbl["dr_start_ticks"].nda[loc] = fcio.event.deadregion[1]
        tbl["dr_stop_pps"].nda[loc] = fcio.event.deadregion[2]
        tbl["dr_stop_ticks"].nda[loc] = fcio.event.deadregion[3]
        tbl["dr_maxticks"].nda[loc] = fcio.event.deadregion[4]
        # the dead-time affected channels
        if fcio.event.deadregion_size >= 7:
            tbl["dr_ch_idx"].nda[loc] = fcio.event.deadregion[5]
            tbl["dr_ch_len"].nda[loc] = fcio.event.deadregion[6]
        else:
            tbl["dr_ch_idx"].nda[loc] = 0
            tbl["dr_ch_len"].nda[loc] = fcio.config.adcs

        # The following values are derived values by fcio-py
        # the time since epoch in seconds
        tbl["timestamp"].nda[loc] = fcio.event.unix_time_utc_sec

        tbl["numtraces"].nda[loc] = fcio.event.num_traces

        start = 0 if loc == 0 else tbl["tracelist"].cumulative_length[loc - 1]
        end = start + fcio.event.num_traces

        tbl["tracelist"].flattened_data.nda[start:end] = fcio.event.trace_list
        tbl["tracelist"].cumulative_length[loc] = end

        tbl["board_id"].flattened_data.nda[start:end] = fcio.event.card_address
        tbl["board_id"].cumulative_length[loc] = end

        tbl["fc_input"].flattened_data.nda[start:end] = fcio.event.card_channel
        tbl["fc_input"].cumulative_length[loc] = end

        tbl["deadinterval_nsec"].flattened_data.nda[
            start:end
        ] = fcio.event.dead_interval_nsec
        tbl["deadinterval_nsec"].cumulative_length[loc] = end

        tbl["deadtime"].flattened_data.nda[start:end] = fcio.event.dead_time_sec
        tbl["deadtime"].cumulative_length[loc] = end

        tbl["deadtime_nsec"].flattened_data.nda[start:end] = fcio.event.dead_time_nsec
        tbl["deadtime_nsec"].cumulative_length[loc] = end

        tbl["lifetime"].flattened_data.nda[start:end] = fcio.event.life_time_sec
        tbl["lifetime"].cumulative_length[loc] = end

        tbl["runtime"].flattened_data.nda[start:end] = fcio.event.run_time_sec
        tbl["runtime"].cumulative_length[loc] = end

        tbl["baseline"].flattened_data.nda[start:end] = fcio.event.fpga_baseline
        tbl["baseline"].cumulative_length[loc] = end

        tbl["daqenergy"].flattened_data.nda[start:end] = fcio.event.fpga_energy
        tbl["daqenergy"].cumulative_length[loc] = end

        evt_hdr_rb.loc += 1

        return evt_hdr_rb.is_full()
