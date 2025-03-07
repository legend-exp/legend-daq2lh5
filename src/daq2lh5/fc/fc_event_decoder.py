from __future__ import annotations

import copy
import logging
from typing import Any

from fcio import FCIO, Tags, Limits
import lgdo

from ..data_decoder import DataDecoder
from .fc_eventheader_decoder import fc_eventheader_decoded_values, get_key

log = logging.getLogger(__name__)

fc_event_decoded_values = copy.deepcopy(fc_eventheader_decoded_values)

# The event decoding splits all arrays in the header into scalars to support decoding per key
for key in [
    "board_id",
    "fc_input",
    "runtime",
    "lifetime",
    "deadtime",
    "deadinterval_nsec",
    "baseline",
    "daqenergy",
]:
    fc_event_decoded_values[key].pop("datatype")
    fc_event_decoded_values[key].pop("length_guess")

# tracelist contains contains the mapping for array indices to channel indices
fc_event_decoded_values.pop("tracelist")
fc_event_decoded_values["channel"] = {
    "dtype": "uint16",
    "description": "The index of each read out channel in this stream.",
}

# Event always carries a waveform
fc_event_decoded_values["waveform"] = {
    "dtype": "uint16",
    "datatype": "waveform",
    "wf_len": Limits.MaxSamples,  # max value. override this before initializing buffers to save RAM
    "dt": 16,  # override if a different clock rate is used
    "dt_units": "ns",
    "t0_units": "ns",
}
"""Default FlashCam Event decoded values.

Re-used by :class:`~.orca.orca_flashcam.ORFlashCamWaveformDecoder`.

Warning
-------
This configuration can be dynamically modified by the decoder at runtime.
"""


class FCEventDecoder(DataDecoder):
    """Decode FlashCam digitizer event data."""

    def __init__(self, *args, **kwargs) -> None:
        self.decoded_values = copy.deepcopy(fc_event_decoded_values)
        super().__init__(*args, **kwargs)
        self.skipped_channels = {}

        # lookup table for the key list, populated in `set_fcio_stream`.
        self.key_list = []
        self.max_rows_in_packet = 0

    def set_fcio_stream(self, fcio_stream: FCIO) -> None:
        """Access ``FCIOConfig`` members once when each file is opened.

        Parameters
        ----------
        fc_config
            extracted via :meth:`~.fc_config_decoder.FCConfigDecoder.decode_config`.
        """

        self.key_list = []
        n_traces = len(fcio_stream.config.tracemap)
        self.max_rows_in_packet = n_traces

        self.decoded_values["waveform"]["wf_len"] = fcio_stream.config.eventsamples
        self.decoded_values["waveform"]["dt"] = fcio_stream.config.sampling_period_ns

        for trace_info in fcio_stream.config.tracemap:
            key = get_key(
                fcio_stream.config.streamid, trace_info >> 16, trace_info & 0xFFFF
            )
            self.key_list.append(key)
        log.debug(
            f"set_file_config: n_traces {n_traces} max_rows {self.max_rows_in_packet} key_list {len(self.key_list)}"
        )

    def get_max_rows_in_packet(self) -> int:
        return self.max_rows_in_packet

    def get_key_lists(self) -> list[list[int | str]]:
        return [copy.deepcopy(self.key_list)]

    def get_decoded_values(self, key: int | str = None) -> dict[str, dict[str, Any]]:
        return self.decoded_values

    def decode_packet(
        self,
        fcio: FCIO,
        evt_rbkd: lgdo.Table | dict[int, lgdo.Table],
        packet_id: int,
    ) -> bool:
        """Access ``FCIOEvent`` members for each event in the DAQ file.

        Parameters
        ----------
        fcio
            The interface to the ``fcio`` data. Enters this function after a
            call to ``fcio.get_record()`` so that data for `packet_id` ready to
            be read out.
        evt_rbkd
            A single table for reading out all data, or a dictionary of tables
            keyed by rawid.
        packet_id
            The index of the packet in the `fcio` stream. Incremented by
            :class:`~.fc.fc_streamer.FCStreamer`.

        Returns
        -------
        any_full
            TODO
        """
        any_full = False

        # a list of channels is read out simultaneously for each event
        for ii, (trace_idx, addr, chan) in enumerate(
            zip(fcio.event.trace_list, fcio.event.card_address, fcio.event.card_channel)
        ):
            key = get_key(fcio.config.streamid, addr, chan)
            if key not in evt_rbkd:
                if key not in self.skipped_channels:
                    # TODO: should this be a warning instead?
                    log.debug(
                        f"skipping packets from channel {trace_idx} index {ii} / key {key}..."
                    )
                    self.skipped_channels[key] = 0
                self.skipped_channels[key] += 1
                continue

            tbl = evt_rbkd[key].lgdo
            loc = evt_rbkd[key].loc

            tbl["channel"].nda[loc] = trace_idx
            tbl["packet_id"].nda[loc] = packet_id
            tbl["fcid"].nda[loc] = fcio.config.streamid & 0xFFFF
            tbl["board_id"].nda[loc] = fcio.event.card_address[ii]
            tbl["fc_input"].nda[loc] = fcio.event.card_channel[ii]
            tbl["event_type"].nda[loc] = fcio.event.type
            tbl["eventnumber"].nda[loc] = fcio.event.timestamp[0]
            tbl["numtraces"].nda[loc] = fcio.event.num_traces
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
            # if event_type == 11: provides the same check
            # however, the usage of deadregion[5]/[6] must never change
            # and it will always be present if deadregion[7..] is ever used
            if fcio.event.deadregion_size >= 7:
                tbl["dr_ch_idx"].nda[loc] = fcio.event.deadregion[5]
                tbl["dr_ch_len"].nda[loc] = fcio.event.deadregion[6]
            else:
                tbl["dr_ch_idx"].nda[loc] = 0
                tbl["dr_ch_len"].nda[loc] = fcio.config.adcs

            # The following values are calculated values by fcio-py
            tbl["timestamp"].nda[loc] = fcio.event.unix_time_utc_sec
            tbl["deadinterval_nsec"].nda[loc] = fcio.event.dead_interval_nsec[ii]
            tbl["deadtime"].nda[loc] = fcio.event.dead_time_sec[ii]
            tbl["lifetime"].nda[loc] = fcio.event.life_time_sec[ii]
            tbl["runtime"].nda[loc] = fcio.event.run_time_sec[ii]
            tbl["baseline"].nda[loc] = fcio.event.fpga_baseline[ii]
            tbl["daqenergy"].nda[loc] = fcio.event.fpga_energy[ii]

            tbl["waveform"]["values"].nda[loc][:] = fcio.event.trace[ii]
            if fcio.config.eventsamples != tbl["waveform"]["values"].nda.shape[1]:
                log.warning(
                    "event wf length was",
                    fcio.config.eventsamples,
                    "when",
                    self.decoded_values["waveform"]["wf_len"],
                    "were expected",
                )

            evt_rbkd[key].loc += 1
            any_full |= evt_rbkd[key].is_full()

        return bool(any_full)
