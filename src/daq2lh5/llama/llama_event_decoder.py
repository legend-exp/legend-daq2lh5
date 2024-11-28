from __future__ import annotations

import copy
import logging
from typing import Any, Dict
import numpy as np

import lgdo

from ..data_decoder import DataDecoder
from .llama_header_decoder import LLAMA_Channel_Configs_t

log = logging.getLogger(__name__)

# put decoded values here
llama_decoded_values_template = {
    # packet index in file
    "packet_id": {"dtype": "uint32"},
    #combined index of FADC and channel
    "fch_id": {"dtype": "uint32"},
    # time since epoch
    "timestamp": {"dtype": "uint64", "units": "clock_ticks"}
    # waveform data --> not always present
    #"waveform": {
    #    "dtype": "uint16",
    #    "datatype": "waveform",
    #    "wf_len": 65532,  # max value. override this before initializing buffers to save RAM
    #    "dt": 8,  # override if a different clock rate is used
    #    "dt_units": "ns",
    #    "t0_units": "ns",
    #}
}
"""Default llamaDAQ SIS3316 Event decoded values.

Warning
-------
This configuration can be dynamically modified by the decoder at runtime.
"""


class LLAMAEventDecoder(DataDecoder):
    """Decode llamaDAQ SIS3316 digitizer event data."""

    def __init__(self, *args, **kwargs) -> None:
        # these are read for every event (decode_event)
        # One set of settings per fch, since settings can be different per channel group
        self.decoded_values: dict[int, dict[str, Any]] = {}
        super().__init__(*args, **kwargs)
        self.skipped_channels = {}      # TODO
        self.channel_configs = None

    def set_channel_configs(self, channel_configs: LLAMA_Channel_Configs_t) -> None:
        """Receive channel configurations from llama_streamer after header was parsed
        Adapt self.decoded_values dict based on read configuration
        """
        self.channel_configs = channel_configs
        for fch, config in self.channel_configs.items():
            self.decoded_values[fch] = copy.deepcopy(llama_decoded_values_template)
            format_bits = config["format_bits"]
            sample_clock_freq = config["sample_freq"]
            avg_mode = config["avg_mode"]
            dt_raw: int = int(1/sample_clock_freq*1000 + 0.5)
            dt_aux: int = dt_raw * (1 << (avg_mode + 1))
            if config["sample_length"] > 0:
                self.__add_waveform(self.decoded_values[fch], False, config["sample_length"], dt_raw)
            if config["avg_sample_length"] > 0 and avg_mode > 0:
                self.__add_waveform(self.decoded_values[fch], True, config["avg_sample_length"], dt_aux)
            if format_bits & 0x01:
                self.__add_accum1till6(self.decoded_values[fch])
            if format_bits & 0x02:
                self.__add_accum7and8(self.decoded_values[fch])
            if format_bits & 0x04:
                self.__add_maw(self.decoded_values[fch])
            if format_bits & 0x08:
                self.__add_energy(self.decoded_values[fch])

    #copied from ORCA SIS3316
    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        if key is None:
            dec_vals_list = self.decoded_values.values()
            if len(dec_vals_list) == 0:
                raise RuntimeError("decoded_values not built yet!")

            return dec_vals_list  # Get first thing we find
        else:
            dec_vals_list = self.decoded_values[key]
            return dec_vals_list
    
    def decode_packet(self, packet: bytes, 
            evt_rbkd: lgdo.Table | dict[int, lgdo.Table],
            packet_id: int,
            fch_id: int
            #header: lgdo.Table | dict[int, lgdo.Table]
            ) -> bool:
        """
        Decodes a single packet, which is a single SIS3316 event, as specified in the Struck manual.
        A single packet corresponds to a single event and channel, and has a unique timestamp.
        packets of different channel groups can vary in size!
        """

        # parse the raw event data into numpy arrays of 16 and 32 bit ints
        evt_data_32 = np.frombuffer(packet, dtype=np.uint32)
        evt_data_16 = np.frombuffer(packet, dtype=np.uint16)

        # e sti gran binaries non ce li metti
        #fch_id = (evt_data_32[0] >> 4) & 0x00000fff  --> to be read earlier, since we need size for chopping out the event from the stream
        timestamp = ((evt_data_32[0] & 0xffff0000) << 16) + evt_data_32[1]
        format_bits = (evt_data_32[0]) & 0x0000000f
        offset = 2
        if format_bits & 0x1:
            peakhigh_value = evt_data_16[4]
            peakhigh_index = evt_data_16[5]
            information = (evt_data_32[offset+1] >> 24) & 0xff
            accumulator1 = evt_data_32[offset+2]
            accumulator2 = evt_data_32[offset+3]
            accumulator3 = evt_data_32[offset+4]
            accumulator4 = evt_data_32[offset+5]
            accumulator5 = evt_data_32[offset+6]
            accumulator6 = evt_data_32[offset+7]
            offset += 7
        else:
            peakhigh_value = 0
            peakhigh_index = 0  
            information = 0
            accumulator1 = accumulator2 = accumulator3 = accumulator4 = accumulator5 = accumulator6 = 0
            pass
        if format_bits & 0x2:
            accumulator7 = evt_data_32[offset+0]
            accumulator8 = evt_data_32[offset+1]
            offset += 2
        else:
            accumulator7 = accumulator8 = 0
            pass
        if format_bits & 0x4:
            mawMax = evt_data_32[offset+0]
            maw_before = evt_data_32[offset+1]
            maw_after = evt_data_32[offset+2]
            offset += 3
        else:
            mawMax = maw_before = maw_after = 0
            pass
        if format_bits & 0x8:
            energy_first = evt_data_32[offset+0]
            energy = evt_data_32[offset+1]
            offset += 2
        else:
            energy_first = energy = 0
            pass
        raw_length_32 = (evt_data_32[offset+0]) & 0x03ffffff
        status_flag = ((evt_data_32[offset+0]) & 0x04000000) >> 26 #bit 26
        maw_test_flag = ((evt_data_32[offset+0]) & 0x08000000) >> 27 #bit 27
        avg_data_coming = False
        if evt_data_32[offset+0] & 0xf0000000 == 0xe0000000:
            avg_data_coming = False
        elif evt_data_32[offset+0] & 0xf0000000 == 0xa0000000:
            avg_data_coming = True
        else:
            raise RuntimeError("Data corruption 1!")
        offset += 1 
        avg_length_32 = 0
        if avg_data_coming:
            avg_count_status = (evt_data_32[offset+0] & 0x00ff0000) >> 16  # bits 23 - 16
            avg_length_32 = evt_data_32[offset+0] & 0x0000ffff
            if evt_data_32[offset+0] & 0xf0000000 != 0xe0000000:
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
            raise RuntimeError("Waveform sizes {} (raw) and {} (avg) doesn't match expected size {}.".format(raw_length_16, avg_length_16, expected_wf_length))

        print(f"Just read event for fch {fch_id}.")
        print(f"ACC1 {accumulator1}, timestamp {timestamp} {raw_length_16}, {avg_length_16}")        

        

        return False


    def __add_waveform(self, decoded_values_fch: dict[str, Any], is_aux: bool, max_samples: int, dt: int) -> None:
        """
        Averaged samples are called "Aux waveform" due to historic (GERDA) reasons.
        """
        name: str = "auxwaveform" if is_aux else "waveform"
        decoded_values_fch[name] = {
            "dtype": "uint16",
            "datatype": "waveform",
            "wf_len": max_samples,  # max value. override this before initializing buffers to save RAM
            "dt": dt,  # override if a different clock rate is used
            "dt_units": "ns",
            "t0_units": "ns",
        }

    def __add_accum1till6(self, decoded_values_fch: dict[str, Any]) -> None:
        decoded_values_fch["peakHighValue"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["peakHighIndex"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["information"] = {"dtype": "uint32"}
        decoded_values_fch["accSum1"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["accSum2"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["accSum3"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["accSum4"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["accSum5"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["accSum6"] = {"dtype": "uint32", "units": "adc"}

    def __add_accum7and8(self, decoded_values_fch: dict[str, Any]) -> None:
        decoded_values_fch["accSum7"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["accSum8"] = {"dtype": "uint32", "units": "adc"}

    def __add_maw(self, decoded_values_fch: dict[str, Any]) -> None:
        decoded_values_fch["mawMax"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["mawBefore"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["mawAfter"] = {"dtype": "uint32", "units": "adc"}

    def __add_energy(self, decoded_values_fch: dict[str, Any]) -> None:
        decoded_values_fch["startEnergy"] = {"dtype": "uint32", "units": "adc"}
        decoded_values_fch["maxEnergy"] = {"dtype": "uint32", "units": "adc"}

