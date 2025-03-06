from __future__ import annotations

import copy
import logging

from fcio import FCIO
import lgdo

from ..data_decoder import DataDecoder
from ..raw_buffer import RawBuffer

log = logging.getLogger(__name__)

# status.data[i].reqid contains the card type in 0xff00 and the index within
# a type group in 0xff
# 0 is readout master
# if bit 0x1000 is set: trigger card
# if bit 0x2000 is set: adc card
# if bit 0x4000 is set: submaster card

fc_status_decoded_values = {
    "packet_id": {"dtype": "uint32", "description": "Packet index in file."},
    "status": {
        "dtype": "bool",
        "description": "True: Ok, False: Errors occurred, check card-wise status.",
    },
    "fpga_time": {
        "dtype": "float32",
        "units": "s",
        "description": "The clock in the fpga of the highest-level card.",
    },
    "server_time": {
        "dtype": "float64",
        "units": "s",
        "description": "The server time when the status was checked.",
    },
    "fpga_start_time": {
        "dtype": "float64",
        "units": "s",
        "description": "The start time of the run.",
    },
    # FC card-wise list DAQ errors during data taking
    "id": {
        "dtype": "uint32",
        "description": "The card id (reqid) when checking the status. Don't confuse with the card address.",
    },
    "eventnumber": {
        "dtype": "uint32",
        "description": "The eventcounter when the status was requested.",
    },
    "fpga_time_nsec": {
        "dtype": "uint32",
        "units": "ns",
        "description": "The clock in the fpga of the highest-level card.",
    },
    "n_total_errors": {
        "dtype": "uint32",
        "description": "The sum of all error counter of all cards.",
    },
    "n_environment_errors": {
        "dtype": "uint32",
        "description": "The sum of all environment sensor errors.",
    },
    "n_cti_errors": {
        "dtype": "uint32",
        "description": "The sum of all CTI connection errors.",
    },
    "n_link_errors": {
        "dtype": "uint32",
        "description": "The sum of all trigger link errors.",
    },
    "n_other_errors": {
        "dtype": "uint32",
        "datatype": "array_of_equalsized_arrays<1,1>{real}",
        "length": 5,
        "description": "The sum of other errors.",
    },
    "mb_temps": {
        "dtype": "int32",
        "units": "mC",
        "datatype": "array_of_equalsized_arrays<1,1>{real}",
        "length": 5,
        "description": "The temperatures measured by sensors on the motherboard.",
    },
    "mb_voltages": {
        "dtype": "int32",
        "units": "mV",
        "datatype": "array_of_equalsized_arrays<1,1>{real}",
        "length": 6,
        "description": "The supply voltages of the motherboard.",
    },
    "mb_current": {
        "dtype": "int32",
        "units": "mA",
        "description": "The current draw of the motherboard.",
    },
    "mb_humidity": {
        "dtype": "int32",
        "units": "o/oo",
        "description": "The humidity in permille measured on the motherboard.",
    },
    "adc_temps": {
        "dtype": "int32",
        "units": "mC",
        "datatype": "array<1>{array<1>{real}}",
        "length_guess": 2,  # max number of daugher cards - only for adc cards
        "description": "If the card has adc daughter (piggy) boards mounted, each daughter has a temperature sensor.",
    },
    "cti_links": {
        "dtype": "uint32",
        "datatype": "array<1>{array<1>{real}}",  # vector of vectors
        "length_guess": 4,
        "description": "CTI debugging values, for experts only.",
    },
    "link_states": {
        "dtype": "uint32",
        "datatype": "array<1>{array<1>{real}}",  # vector of vectors
        "length_guess": 256,  # 256*max number of boards
        "description": "Trigger link debugging values, for experts only.",
    },
}


def get_key(streamid, reqid):
    return (streamid & 0xFFFF) * 1000000 + (reqid & 0xFFFF)

def get_fcid(key: int) -> int:
    return int(key // 10000000)

def get_reqid(key: int) -> int:
    return int(key % 1000000)


class FCStatusDecoder(DataDecoder):
    """Decode FlashCam digitizer status data."""

    def __init__(self, *args, **kwargs) -> None:
        self.key_list = []
        self.max_rows_in_packet = 0
        self.decoded_values = copy.deepcopy(fc_status_decoded_values)
        """Default FlashCam status decoded values.

        Warning
        -------
        This configuration can be dynamically modified by the decoder at runtime.
        """
        super().__init__(*args, **kwargs)

    def set_fcio_stream(self, fcio_stream: FCIO) -> None:
        """Access ``FCIOConfig`` members once when each file is opened.

        Parameters
        ----------
        fcio_stream
            extracted via :meth:`~.fc_config_decoder.FCConfigDecoder.decode_config`.
        """

        n_cards = (
            fcio_stream.config.mastercards
            + fcio_stream.config.triggercards
            + fcio_stream.config.adccards
        )
        self.max_rows_in_packet = n_cards

        # the number of master cards is the sum of top and sub masters,
        # if there is more than one, the first is the top master
        # the rest is a sub master card.
        for i in range(fcio_stream.config.mastercards):
            if i == 0:
                key = get_key(fcio_stream.config.streamid, 0)
            else:
                key = get_key(fcio_stream.config.streamid, 0x4000 + i - 1)
            self.key_list.append(key)

        for i in range(fcio_stream.config.triggercards):
            key = get_key(fcio_stream.config.streamid, 0x1000 + i)
            self.key_list.append(key)

        for i in range(fcio_stream.config.adccards):
            key = get_key(fcio_stream.config.streamid, 0x2000 + i)
            self.key_list.append(key)

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
        status_rbkd: lgdo.Table | dict[int, lgdo.Table],
        packet_id: int,
    ) -> bool:
        """Access ``FCIOStatus`` members for each status packet in the DAQ file.

        Parameters
        ----------
        fcio
            The interface to the ``fcio`` data. Enters this function after a
            call to ``fcio.get_record()`` so that data for `packet_id` ready to
            be read out.
        status_rbkd
            A single table for reading out all data, or a dictionary of tables
            keyed by card number.
        packet_id
            The index of the packet in the `fcio` stream. Incremented by
            :class:`~.fc.fc_streamer.FCStreamer`.

        Returns
        -------
        any_full
            TODO
        """
        any_full = False
        for card_data in fcio.status.data:
            key = get_key(fcio.config.streamid, card_data.reqid)
            if key not in status_rbkd:
                continue

            tbl = status_rbkd[key].lgdo
            loc = status_rbkd[key].loc

            tbl["packet_id"].nda[loc] = packet_id
            tbl["status"].nda[loc] = fcio.status.status

            # times
            tbl["fpga_time"].nda[loc] = fcio.status.fpga_time_sec
            tbl["server_time"].nda[loc] = fcio.status.unix_time_utc_sec
            tbl["fpga_start_time"].nda[loc] = fcio.status.fpga_start_time_sec

            # per card information
            tbl["id"].nda[loc] = card_data.reqid
            tbl["eventnumber"].nda[loc] = card_data.eventno
            tbl["fpga_time_nsec"].nda[loc] = card_data.fpga_time_nsec
            tbl["n_total_errors"].nda[loc] = card_data.totalerrors
            tbl["n_environment_errors"].nda[loc] = card_data.enverrors
            tbl["n_cti_errors"].nda[loc] = card_data.ctierrors
            tbl["n_other_errors"].nda[loc][:] = card_data.othererrors
            tbl["mb_temps"].nda[loc][:] = card_data.mainboard_temperatures_mC
            tbl["mb_voltages"].nda[loc][:] = card_data.mainboard_voltages_mV
            tbl["mb_current"].nda[loc] = card_data.mainboard_current_mA
            tbl["mb_humidity"].nda[loc] = card_data.mainboard_humiditiy_permille
            tbl["adc_temps"]._set_vector_unsafe(
                loc, card_data.daughterboard_temperatures_mC
            )
            tbl["cti_links"]._set_vector_unsafe(loc, card_data.ctilinks)
            tbl["link_states"]._set_vector_unsafe(loc, card_data.linkstates)

            status_rbkd[key].loc += 1
            any_full |= status_rbkd[key].is_full()

        return bool(any_full)
