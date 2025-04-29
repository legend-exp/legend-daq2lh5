from __future__ import annotations

import copy
import logging
from typing import Any

import lgdo
import numpy as np

from ..data_decoder import DataDecoder
from .llama_header_decoder import LLAMA_Channel_Configs_t

log = logging.getLogger(__name__)

# put decoded values here
llama_decoded_values_template = {
    # packet index in file
    "packet_id": {"dtype": "uint32"},
    # combined index of FADC and channel
    "fadc_channel_id": {"dtype": "uint32"},
    # time since epoch
    "time_since_run_start": {"dtype": "float64", "units": "s"},
    "unixtime": {"dtype": "float64", "units": "s"},
    "unixtime_accuracy": {"dtype": "float64", "units": "s"},
    "fadc_status_bits": {"dtype": "uint32"},
    # waveform data --> not always present
    # "waveform": {
    #    "dtype": "uint16",
    #    "datatype": "waveform",
    #    "wf_len": 65532,  # max value. override this before initializing buffers to save RAM
    #    "dt": 8,  # override if a different clock rate is used
    #    "dt_units": "ns",
    #    "t0_units": "ns",
    # }
}
# """Default llamaDAQ SIS3316 Event decoded values.
#
# Warning
# -------
# This configuration will be dynamically modified by the decoder at runtime.
# """


def check_dict_spec_equal(
    d1: dict[str, Any], d2: dict[str, Any], specs: list[str]
) -> bool:
    for spec in specs:
        if d1.get(spec) != d2.get(spec):
            return False
    return True


class LLAMAEventDecoder(DataDecoder):
    """Decode llamaDAQ SIS3316 digitizer event data."""

    def __init__(self, *args, **kwargs) -> None:
        # these are read for every event (decode_event)
        # One set of settings per fch, since settings can be different per channel group
        self.decoded_values: dict[int, dict[str, Any]] = {}
        super().__init__(*args, **kwargs)
        self.skipped_channels = {}
        self.channel_configs = None
        self.dt_raw: dict[int, float] = (
            {}
        )  # need to buffer that to update t0 for avg waveforms per event
        self.t0_raw: dict[int, float] = (
            {}
        )  # store when receiving channel configs and use for each waveform
        self.t0_avg_const: dict[int, float] = (
            {}
        )  # constant part of the t0 of averaged waveforms

    def set_channel_configs(self, channel_configs: LLAMA_Channel_Configs_t) -> None:
        """Receive channel configurations from llama_streamer after header was parsed.

        Adapt self.decoded_values dict based on read configuration.
        """
        self.channel_configs = channel_configs
        for fch, config in self.channel_configs.items():
            self.decoded_values[fch] = copy.deepcopy(llama_decoded_values_template)
            format_bits = config["format_bits"]
            sample_clock_freq = config["sample_freq"]
            avg_mode = config["avg_mode"]
            dt_raw: float = 1 / sample_clock_freq * 1000
            dt_avg: float = dt_raw * (1 << (avg_mode + 1))
            # t0 generation functions from llamaDAQ -> EventConfig.hh
            t0_raw: float = (
                float(config["sample_start_index"]) - float(config["sample_pretrigger"])
            ) * dt_raw  # location of the trigger is at t = 0
            t0_avg: float = (
                -float(config["sample_pretrigger"]) * float(dt_raw)
                - float(config["avg_sample_pretrigger"]) * dt_avg
            )  # additional offset to be added independently for every event
            self.dt_raw[fch] = dt_raw
            self.t0_raw[fch] = t0_raw
            self.t0_avg_const[fch] = t0_avg
            if config["sample_length"] > 0:
                self.__add_waveform(
                    self.decoded_values[fch], False, config["sample_length"], dt_raw
                )
            if config["avg_sample_length"] > 0 and avg_mode > 0:
                self.__add_waveform(
                    self.decoded_values[fch], True, config["avg_sample_length"], dt_avg
                )
            if format_bits & 0x01:
                self.__add_accum1till6(self.decoded_values[fch])
            if format_bits & 0x02:
                self.__add_accum7and8(self.decoded_values[fch])
            if format_bits & 0x04:
                self.__add_maw(self.decoded_values[fch])
            if format_bits & 0x08:
                self.__add_energy(self.decoded_values[fch])

    def set_global_configs(self, global_configs: dict[str, Any]):
        self.global_configs = global_configs

    def get_key_lists(self) -> list[list[int | str]]:
        """Get list of keys.

        Return a list of lists of keys available for this decoder.  Each inner
        list are the fadc_channel_id's which share the exact same settings
        (trace lengths, avg mode, ...), so they can end up in the same buffer.
        """
        if self.channel_configs is None:
            raise RuntimeError(
                "Identification of key lists requires channel configs to be set!"
            )

        params_for_equality = ["sample_length", "avg_sample_length", "avg_mode"]

        def check_equal(c1, c2):
            return check_dict_spec_equal(c1, c2, params_for_equality)

        kll: list[list[int]] = []  # key-list-list
        for fadc_channel_id, config in self.channel_configs.items():
            for kl in kll:
                # use 1st entry of a list of list as "archetype"
                if check_equal(config, self.channel_configs[kl[0]]):
                    kl.append(fadc_channel_id)
                    break
            else:
                kll.append([fadc_channel_id])
        log.debug(f"key lists are: {repr(kll)}")
        return kll

    # copied from ORCA SIS3316
    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        if key is None:
            raise RuntimeError("Key is None!")
            dec_vals_list = self.decoded_values.values()
            if len(dec_vals_list) == 0:
                raise RuntimeError("decoded_values not built yet!")

            return dec_vals_list  # Get first thing we find
        else:
            dec_vals_list = self.decoded_values[key]
            return dec_vals_list

    def decode_packet(
        self,
        packet: bytes,
        evt_rbkd: lgdo.Table | dict[int, lgdo.Table],
        packet_id: int,
        fadc_channel_id: int,
        # header: lgdo.Table | dict[int, lgdo.Table]
    ) -> bool:
        """Decode packet.

        Decodes a single packet, which is a single SIS3316 event, as specified
        in the Struck manual.  A single packet corresponds to a single event
        and channel, and has a unique timestamp.  packets of different channel
        groups can vary in size!
        """

        # Check if this fadc_channel_id should be recorded.
        if fadc_channel_id not in evt_rbkd:
            if fadc_channel_id not in self.skipped_channels:
                self.skipped_channels[fadc_channel_id] = 0
                log.info(f"Skipping channel: {fadc_channel_id}")
                log.debug(f"evt_rbkd: {evt_rbkd.keys()}")
            self.skipped_channels[fadc_channel_id] += 1
            return False

        tbl = evt_rbkd[fadc_channel_id].lgdo
        ii = evt_rbkd[fadc_channel_id].loc

        # parse the raw event data into numpy arrays of 16 and 32 bit ints
        evt_data_32 = np.frombuffer(packet, dtype=np.uint32)
        evt_data_16 = np.frombuffer(packet, dtype=np.uint16)

        # e sti gran binaries non ce li metti
        # fadc_channel_id = (evt_data_32[0] >> 4) & 0x00000fff  --> to be read earlier,
        # since we need size for chopping out the event from the stream
        timestamp_in_clock_ticks = ((evt_data_32[0] & 0xFFFF0000) << 16) + evt_data_32[
            1
        ]
        format_bits = (evt_data_32[0]) & 0x0000000F

        timestamp = timestamp_in_clock_ticks * self.dt_raw[fadc_channel_id] * 1e-9

        tbl["fadc_channel_id"].nda[ii] = fadc_channel_id
        tbl["packet_id"].nda[ii] = packet_id
        tbl["time_since_run_start"].nda[ii] = timestamp
        tbl["unixtime"].nda[ii] = timestamp + self.global_configs["initial_timestamp"]
        # accuracy when comparing timestamps of different files:
        tbl["unixtime_accuracy"].nda[ii] = self.global_configs[
            "initial_timestamp_accuracy"
        ]

        offset = 2
        if format_bits & 0x1:
            tbl["wf_max_sample_value"].nda[ii] = evt_data_16[4]
            tbl["wf_max_sample_idx"].nda[ii] = evt_data_16[5]
            tbl["info_bits"].nda[ii] = (evt_data_32[offset + 1] >> 24) & 0xFF
            tbl["cumsum_1"].nda[ii] = evt_data_32[offset + 2]
            tbl["cumsum_2"].nda[ii] = evt_data_32[offset + 3]
            tbl["cumsum_3"].nda[ii] = evt_data_32[offset + 4]
            tbl["cumsum_4"].nda[ii] = evt_data_32[offset + 5]
            tbl["cumsum_5"].nda[ii] = evt_data_32[offset + 6]
            tbl["cumsum_6"].nda[ii] = evt_data_32[offset + 7]
            offset += 7

        if format_bits & 0x2:
            tbl["cumsum_7"].nda[ii] = evt_data_32[offset + 0]
            tbl["cumsum_8"].nda[ii] = evt_data_32[offset + 1]
            offset += 2

        if format_bits & 0x4:
            tbl["maw_max"].nda[ii] = evt_data_32[offset + 0]
            tbl["maw_before"].nda[ii] = evt_data_32[offset + 1]
            tbl["maw_after"].nda[ii] = evt_data_32[offset + 2]
            offset += 3

        if format_bits & 0x8:
            tbl["start_energy"].nda[ii] = evt_data_32[offset + 0]
            tbl["max_energy"].nda[ii] = evt_data_32[offset + 1]
            offset += 2

        raw_length_32 = (evt_data_32[offset + 0]) & 0x03FFFFFF
        tbl["fadc_status_bits"].nda[ii] = (
            (evt_data_32[offset + 0]) & 0x04000000
        ) >> 26  # bit 26
        maw_test_flag = ((evt_data_32[offset + 0]) & 0x08000000) >> 27  # bit 27
        avg_data_coming = False
        if evt_data_32[offset + 0] & 0xF0000000 == 0xE0000000:
            avg_data_coming = False
        elif evt_data_32[offset + 0] & 0xF0000000 == 0xA0000000:
            avg_data_coming = True
        else:
            raise RuntimeError("Data corruption 1!")
        offset += 1
        avg_length_32 = 0
        if avg_data_coming:
            avg_count_status = (
                evt_data_32[offset + 0] & 0x00FF0000
            ) >> 16  # bits 23 - 16
            avg_length_32 = evt_data_32[offset + 0] & 0x0000FFFF
            if evt_data_32[offset + 0] & 0xF0000000 != 0xE0000000:
                raise RuntimeError("Data corruption 2!")
            offset += 1

        # --- now the offset points to the raw wf data ---

        if maw_test_flag:
            raise RuntimeError("Cannot handle data with MAW test data!")

        # compute expected and actual array dimensions
        raw_length_16 = 2 * raw_length_32
        avg_length_16 = 2 * avg_length_32
        header_length_16 = offset * 2
        expected_wf_length = len(evt_data_16) - header_length_16

        # error check: waveform size must match expectations
        if raw_length_16 + avg_length_16 != expected_wf_length:
            raise RuntimeError(
                f"Waveform sizes {raw_length_16} (raw) and {avg_length_16} (avg) "
                f"doesn't match expected size {expected_wf_length}."
            )

        # store waveform if available:
        if raw_length_16 > 0:
            tbl["waveform_windowed"]["values"].nda[ii] = evt_data_16[
                offset * 2 : offset * 2 + raw_length_16
            ]
            offset += raw_length_32
            tbl["waveform_windowed"]["t0"].nda[ii] = self.t0_raw[fadc_channel_id]

        # store pre-averaged (avg) waveform if available:
        if avg_length_16 > 0:
            tbl["waveform_presummed"]["values"].nda[ii] = evt_data_16[
                offset * 2 : offset * 2 + avg_length_16
            ]
            offset += avg_length_32
            # need to update avg waveform t0 based on the offset I get per event
            tbl["waveform_presummed"]["t0"].nda[ii] = (
                self.t0_avg_const[fadc_channel_id]
                + float(avg_count_status) * self.dt_raw[fadc_channel_id]
            )

        if offset != len(evt_data_32):
            raise RuntimeError("I messed up...")

        evt_rbkd[fadc_channel_id].loc += 1

        return evt_rbkd[fadc_channel_id].is_full()

    def __add_waveform(
        self,
        decoded_values_fch: dict[str, Any],
        is_avg: bool,
        max_samples: int,
        dt: float,
    ) -> None:
        """Helper function to add waveform.

        Averaged samples are available from the 125 MHz (16 bit) variatnt of
        the SIS3316 and can be stored independently of raw samples.  I use
        waveform for raw samples (dt from clock itself) and waveform_presummed
        from averaged samples (dt from clock * avg number).

        GERDA used to have the low-frequency (waveform) & the high-frequency
        (aux waveform); here: LF = waveform_presummed & HF = waveform_windowed.
        """
        name: str = "waveform_presummed" if is_avg else "waveform_windowed"
        decoded_values_fch[name] = {
            "dtype": "uint16",
            "datatype": "waveform",
            "wf_len": max_samples,  # max value. override this before initializing buffers to save RAM
            "dt": dt,  # the sample pitch (inverse of clock speed)
            # "t0": t0, # Adding t0 here does not work
            "dt_units": "ns",
            "t0_units": "ns",
        }

    def __add_accum1till6(self, decoded_values_fch: dict[str, Any]) -> None:
        decoded_values_fch["wf_max_sample_value"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["wf_max_sample_idx"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["info_bits"] = {"dtype": "uint32"}
        decoded_values_fch["cumsum_1"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["cumsum_2"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["cumsum_3"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["cumsum_4"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["cumsum_5"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["cumsum_6"] = {"dtype": "uint32", "units": "adc"}

    def __add_accum7and8(self, decoded_values_fch: dict[str, Any]) -> None:
        decoded_values_fch["cumsum_7"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["cumsum_8"] = {"dtype": "uint32", "units": "adc"}

    def __add_maw(self, decoded_values_fch: dict[str, Any]) -> None:
        decoded_values_fch["maw_max"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["maw_before"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["maw_after"] = {"dtype": "uint32", "units": "adc"}

    def __add_energy(self, decoded_values_fch: dict[str, Any]) -> None:
        decoded_values_fch["start_energy"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["max_energy"] = {"dtype": "uint32", "units": "adc"}
