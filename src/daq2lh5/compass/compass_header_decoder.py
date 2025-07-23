from __future__ import annotations

import logging

import lgdo
import numpy as np

from ..data_decoder import DataDecoder
from .compass_config_parser import compass_config_to_struct

log = logging.getLogger(__name__)


class CompassHeaderDecoder(DataDecoder):
    """
    Decode CoMPASS header data. Also, read in CoMPASS config data if provided using the compass_config_parser
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = None  # initialize to none, because compass_config_to_struct always returns a struct

    def decode_header(self, in_stream: bytes, config_file: str = None) -> dict:
        """Decode the CoMPASS file header, and add CoMPASS config data to the header, if present.

        Parameters
        ----------
        in_stream
            The stream of data to have its header decoded
        config_file
            The config file for the CoMPASS data, if present

        Returns
        -------
        config
            A dict containing the header information, as well as the important config information
            of wf_len and num_enabled_channels
        """
        wf_len = None
        config_names = [
            "energy_channels",  # energy is given in channels (0: false, 1: true)
            "energy_calibrated",  # energy is given in keV/MeV, according to the calibration (0: false, 1: true)
            "energy_short",  # energy short is present (0: false, 1: true)
            "waveform_samples",  # waveform samples are present (0: false, 1: true)
            "header_present",  # there is a 2 byte header present in the file (0: false, 1: true)
        ]  # need to determine which of these are present in a file

        # First need to check if the first two bytes are of the form 0xCAEx
        # CoMPASS specs say that every file should start with this header, but if CoMPASS writes size-limited files, then this header may not be present in *all* files...
        header_in_bytes = in_stream.read(2)

        if header_in_bytes[-1] == int.from_bytes(b"\xca", byteorder="big"):
            log.debug("header is present in file.")
            header_in_binary = bin(int.from_bytes(header_in_bytes, byteorder="little"))
            header_as_list = str(header_in_binary)[
                ::-1
            ]  # reverse it as we care about bit 0, bit 1, etc.
            header_dict = dict(
                {
                    "energy_channels": int(header_as_list[0]) == 1,
                    "energy_calibrated": int(header_as_list[1]) == 1,
                    "energy_channels_calibrated": int(header_as_list[0])
                    == 1 & int(header_as_list[1])
                    == 1,
                    "energy_short": int(header_as_list[2]) == 1,
                    "waveform_samples": int(header_as_list[3]) == 1,
                    "header_present": True,
                }
            )

            # if we don't have the wf_len, get it now

            if config_file is None:
                if header_dict["waveform_samples"] == 0:
                    wf_len = 0
                else:
                    wf_byte_len = 4
                    bytes_to_read = (
                        12  # covers 2-byte board, 2-byte channel, 8-byte time stamp
                    )
                    bytes_to_read += (
                        2 * header_dict["energy_channels"]
                        + 8 * header_dict["energy_calibrated"]
                        + 2 * header_dict["energy_short"]
                    )
                    bytes_to_read += 4 + 1  # 4-byte flags, 1-byte waveform code
                    first_bytes = in_stream.read(bytes_to_read + wf_byte_len)

                    wf_len = np.frombuffer(
                        first_bytes[bytes_to_read : bytes_to_read + wf_byte_len],
                        dtype=np.uint32,
                    )[0]

        # if header is not present, we need to play some tricks
        # either energy short is present or not
        # and one of three options for energy (ADC, calibrated, both)
        else:
            # If the 2 byte header is not present, then we have read in the board by accident
            header_in_bytes += in_stream.read(
                10
            )  # read in the 2-byte ch and 8-byte timestamp
            bytes_read = 12
            fixed_header_start_len = (
                12  # always 12 bytes: 2-byte board, 2-byte channel, 8-byte timestamp
            )
            possible_energy_header_byte_lengths = [
                2,
                8,
                10,
            ]  # either ADC, Calibrated, or both
            possible_energy_short_header_byte_lengths = [
                0,
                2,
            ]  # energy short is present or not
            fixed_header_part = 5  # 5 bytes from flags + code
            wf_len_bytes = 4  # wf_len is 4 bytes long

            for prefix in possible_energy_header_byte_lengths:
                terminate = False
                for suffix in possible_energy_short_header_byte_lengths:

                    # ---- first packet -------
                    # don't read more than we have to, check how many more bytes we need to read in
                    difference = (
                        fixed_header_start_len
                        + prefix
                        + suffix
                        + fixed_header_part
                        + wf_len_bytes
                        - bytes_read
                    )
                    if difference > 0:
                        # just read a bit more data
                        header_in_bytes += in_stream.read(difference)
                        bytes_read += difference

                    wf_len_guess = np.frombuffer(
                        header_in_bytes[
                            fixed_header_start_len
                            + prefix
                            + suffix
                            + fixed_header_part : fixed_header_start_len
                            + prefix
                            + suffix
                            + fixed_header_part
                            + wf_len_bytes
                        ],
                        dtype=np.uint32,
                    )[0]

                    # read in the first waveform data
                    difference = (
                        fixed_header_start_len
                        + prefix
                        + suffix
                        + fixed_header_part
                        + wf_len_bytes
                        + 2 * wf_len_guess
                        - bytes_read
                    )
                    if difference > 0:
                        header_in_bytes += in_stream.read(2 * wf_len_guess)
                        bytes_read += 2 * wf_len_guess

                    # ------ second packet header ----------
                    difference = (
                        2
                        * (
                            fixed_header_start_len
                            + prefix
                            + suffix
                            + fixed_header_part
                            + wf_len_bytes
                        )
                        + 2 * wf_len_guess
                        - bytes_read
                    )
                    if difference > 0:
                        header_in_bytes += in_stream.read(difference)
                        bytes_read += (
                            fixed_header_start_len
                            + prefix
                            + suffix
                            + fixed_header_part
                            + wf_len_bytes
                        )
                    wf_len_guess_2 = np.frombuffer(
                        header_in_bytes[
                            2
                            * (
                                fixed_header_start_len
                                + prefix
                                + suffix
                                + fixed_header_part
                            )
                            + wf_len_bytes
                            + 2
                            * wf_len_guess : 2
                            * (
                                fixed_header_start_len
                                + prefix
                                + suffix
                                + fixed_header_part
                                + wf_len_bytes
                            )
                            + 2 * wf_len_guess
                        ],
                        dtype=np.uint32,
                    )[0]

                    # if the waveform lengths agree, then we can stride packets correctly
                    if wf_len_guess_2 == wf_len_guess:
                        header_dict = dict(
                            {
                                "energy_channels": prefix == 2,
                                "energy_calibrated": prefix == 8,
                                "energy_channels_calibrated": prefix == 10,
                                "energy_short": suffix == 2,
                                "waveform_samples": wf_len != 0,
                                "header_present": False,
                            }
                        )
                        wf_len = wf_len_guess
                        terminate = True
                        break
                if terminate:
                    break

        self.config = compass_config_to_struct(config_file, wf_len)

        for name in config_names:
            if name in self.config:
                log.warning(f"{name} already in self.config. skipping...")
                continue
            value = int(header_dict[name])
            self.config.add_field(
                str(name), lgdo.Scalar(value)
            )  # self.config is a struct

        return self.config

    def make_lgdo(self, key: int = None, size: int = None) -> lgdo.Struct:
        if self.config is None:
            raise RuntimeError(
                "self.config still None, need to decode header before calling make_lgdo"
            )
        return self.config  # self.config is already an lgdo, namely it is a struct
