from __future__ import annotations

import datetime
import io
import logging
import os
import re
from typing import Any, Dict

import lgdo
import numpy as np

from ..data_decoder import DataDecoder
from .llama_base import join_fadcid_chid

log = logging.getLogger(__name__)

LLAMA_Channel_Configs_t = Dict[int, Dict[str, Any]]


def parse_filename_for_timestamp(f_in_name: str) -> float:
    """take a filename; return the unixtime parsed from the filename; 0 if impossible."""
    filename = os.path.basename(f_in_name)
    if match := re.fullmatch(r".*(\d{8})[-T](\d{6})(Z?).*", filename):
        tsymd = match.group(1)
        tshms = match.group(2)
        utc: bool = True if match.group(3) == "Z" else False
        when_file: datetime.datetime = datetime.datetime.strptime(
            tsymd + tshms, "%Y%m%d%H%M%S"
        )  # naive datetime object
        if utc:
            when_file = when_file.replace(
                tzinfo=datetime.timezone.utc
            )  # make it aware; UTC. Naive is treated as local
        return when_file.timestamp()
    else:
        return 0


class LLAMAHeaderDecoder(DataDecoder):  # DataDecoder currently unused
    """Decode llamaDAQ header data.

    Includes the file header as well as all available ("open") channel
    configurations.
    """

    @staticmethod
    def magic_bytes() -> int:
        return 0x4972414C

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = lgdo.Struct()
        self.channel_configs = None

    def decode_header(self, f_in: io.BufferedReader, f_in_name: str) -> lgdo.Struct:
        n_bytes_read = 0

        f_in.seek(0)  # should be there anyhow, but re-set if not
        header = f_in.read(16)  # read 16 bytes
        n_bytes_read += 16
        evt_data_32 = np.frombuffer(header, dtype=np.uint32)
        evt_data_16 = np.frombuffer(header, dtype=np.uint16)

        # line0: magic bytes
        magic = evt_data_32[0]
        # print(hex(magic))
        if magic == self.magic_bytes():
            log.info("Read in file as llamaDAQ-SIS3316, magic bytes correct.")
        else:
            log.error("Magic bytes not matching for llamaDAQ file!")
            raise RuntimeError("wrong file type")

        self.version_major = evt_data_16[4]
        self.version_minor = evt_data_16[3]
        self.version_patch = evt_data_16[2]
        self.length_econf = evt_data_16[5]
        self.number_chOpen = evt_data_32[3]

        self.global_configs = {}

        # currently pulled from filename with 1s precision.
        # change if we have it in the llamaDAQ file's header
        self.global_configs["initial_timestamp"] = parse_filename_for_timestamp(
            f_in_name
        )
        self.global_configs["initial_timestamp_accuracy"] = 1.0  # in seconds

        log.debug(
            f"File version: {self.version_major}.{self.version_minor}.{self.version_patch}"
        )
        log.debug(
            f"{self.number_chOpen} channels open, each config {self.length_econf} bytes long"
        )
        n_bytes_read += self.__decode_channel_configs(f_in)

        # print(self.channel_configs[0]["maw3_offset"])

        # assemble LGDO struct:
        self.config.add_field("version_major", lgdo.Scalar(self.version_major))
        self.config.add_field("version_minor", lgdo.Scalar(self.version_minor))
        self.config.add_field("version_patch", lgdo.Scalar(self.version_patch))
        self.config.add_field("length_econf", lgdo.Scalar(self.length_econf))
        self.config.add_field("number_chOpen", lgdo.Scalar(self.number_chOpen))
        self.config.add_field(
            "initial_timestamp", lgdo.Scalar(self.global_configs["initial_timestamp"])
        )
        self.config.add_field(
            "initial_timestamp_accuracy",
            lgdo.Scalar(self.global_configs["initial_timestamp_accuracy"]),
        )

        for fch_id, fch_content in self.channel_configs.items():
            fch_lgdo = lgdo.Struct()
            for key, value in fch_content.items():
                fch_lgdo.add_field(key, lgdo.Scalar(value))
            self.config.add_field(f"fch_{fch_id:02d}", fch_lgdo)

        return self.config, n_bytes_read

    # override from DataDecoder
    def make_lgdo(self, key: int = None, size: int = None) -> lgdo.Struct:
        return self.config

    def get_channel_configs(self) -> LLAMA_Channel_Configs_t:
        return self.channel_configs

    def get_global_configs(self) -> dict[str, Any]:
        return self.global_configs

    def __decode_channel_configs(self, f_in: io.BufferedReader) -> int:
        """Reads the metadata.

        Reads the metadata from the beginning of the file (the "channel
        configuration" part, directly after the file header).  Creates a
        dictionary of the metadata for each FADC/channel combination, which is
        returned. Returns number of bytes read.

        FADC-ID and channel-ID are combined into a single id for flattening:
        ``(fadcid << 4) + chid``.
        """
        # f_in.seek(16)    #should be after file header anyhow, but re-set if not
        n_bytes_read = 0
        self.channel_configs = {}

        if self.length_econf != 88:
            raise RuntimeError("Invalid channel configuration format")

        for _i in range(0, self.number_chOpen):
            # print("reading in channel config {}".format(i))

            channel = f_in.read(self.length_econf)
            n_bytes_read += int(self.length_econf)
            ch_dpf = channel[16:32]
            evt_data_32 = np.frombuffer(channel, dtype=np.uint32)
            evt_data_dpf = np.frombuffer(ch_dpf, dtype=np.float64)

            fadc_index = evt_data_32[0]
            channel_index = evt_data_32[1]
            fch_id = join_fadcid_chid(fadc_index, channel_index)

            if fch_id in self.channel_configs:
                raise RuntimeError(
                    f"duplicate channel configuration in file: FADCID: "
                    f"{fadc_index}, ChannelID: {channel_index}"
                )
            else:
                self.channel_configs[fch_id] = {}

            self.channel_configs[fch_id]["14_bit_flag"] = evt_data_32[2] & 0x00000001
            if evt_data_32[2] & 0x00000002 == 0:
                log.warning("Channel in configuration marked as non-open!")
            self.channel_configs[fch_id]["adc_offset"] = evt_data_32[3]
            self.channel_configs[fch_id]["sample_freq"] = evt_data_dpf[
                0
            ]  # 64 bit float
            self.channel_configs[fch_id]["gain"] = evt_data_dpf[1]
            self.channel_configs[fch_id]["format_bits"] = evt_data_32[8]
            self.channel_configs[fch_id]["sample_start_index"] = evt_data_32[9]
            self.channel_configs[fch_id]["sample_pretrigger"] = evt_data_32[10]
            self.channel_configs[fch_id]["avg_sample_pretrigger"] = evt_data_32[11]
            self.channel_configs[fch_id]["avg_mode"] = evt_data_32[12]
            self.channel_configs[fch_id]["sample_length"] = evt_data_32[13]
            self.channel_configs[fch_id]["avg_sample_length"] = evt_data_32[14]
            self.channel_configs[fch_id]["maw_buffer_length"] = evt_data_32[15]
            self.channel_configs[fch_id]["event_length"] = evt_data_32[16]
            self.channel_configs[fch_id]["event_header_length"] = evt_data_32[17]
            self.channel_configs[fch_id]["accum6_offset"] = evt_data_32[18]
            self.channel_configs[fch_id]["accum2_offset"] = evt_data_32[19]
            self.channel_configs[fch_id]["maw3_offset"] = evt_data_32[20]
            self.channel_configs[fch_id]["energy_offset"] = evt_data_32[21]

        return n_bytes_read
