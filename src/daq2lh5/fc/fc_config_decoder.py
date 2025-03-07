from __future__ import annotations

import copy
import logging

from fcio import FCIO, Limits
import lgdo
import numpy as np

from daq2lh5.raw_buffer import RawBufferList

from ..data_decoder import DataDecoder

log = logging.getLogger(__name__)

fc_config_decoded_values = {
    "packet_id": {
        "dtype": "uint32",
        "description": "The index of this decoded packet in the file.",
    },
    "nsamples": {"dtype": "int32", "description": "samples per channel"},
    "nadcs": {"dtype": "int32", "description": "number of adc channels"},
    "ntriggers": {"dtype": "int32", "description": "number of triggertraces"},
    "streamid": {"dtype": "int32", "description": "id of stream"},
    "adcbits": {"dtype": "int32", "description": "bit range of the adc channels"},
    "sumlength": {"dtype": "int32", "description": "length of the fpga integrator"},
    "blprecision": {"dtype": "int32", "description": "precision of the fpga baseline"},
    "mastercards": {"dtype": "int32", "description": "number of attached mastercards"},
    "triggercards": {
        "dtype": "int32",
        "description": "number of attached triggercards",
    },
    "adccards": {"dtype": "int32", "description": "number of attached fadccards"},
    "gps": {
        "dtype": "int32",
        "description": "gps mode (0: not used, >0: external pps and 10MHz)",
    },
    "tracemap": {
        "dtype": "uint32",
        "datatype": "array<1>{array<1>{real}}",
        "length": Limits.MaxChannels,
        "description": "",
    },
}


class FCConfigDecoder(DataDecoder):
    """Decode FlashCam config data.

    Note
    ----
    Derives from :class:`~.data_decoder.DataDecoder` in anticipation of
    possible future functionality. Currently the base class interface is not
    used.

    Example
    -------
    >>> from fcio import fcio_open
    >>> from daq2lh5.fc.config_decoder import FCConfigDecoder
    >>> fc = fcio_open('file.fcio')
    >>> decoder = FCConfigDecoder()
    >>> config = decoder.decode_config(fc)
    >>> type(config)
    lgdo.types.struct.Struct
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoded_values = copy.deepcopy(fc_config_decoded_values)

    def decode_packet(
        self,
        fcio: FCIO,
        config_rb: lgdo.Table,
        packet_id: int,
    ) -> bool:

        tbl = config_rb.lgdo
        loc = config_rb.loc

        tbl["packet_id"].nda[loc] = packet_id

        tbl["nsamples"].nda[loc] = fcio.config.eventsamples
        tbl["nadcs"].nda[loc] = fcio.config.adcs
        tbl["ntriggers"].nda[loc] = fcio.config.triggers
        tbl["streamid"].nda[loc] = fcio.config.streamid
        tbl["adcbits"].nda[loc] = fcio.config.adcbits
        tbl["sumlength"].nda[loc] = fcio.config.sumlength
        tbl["blprecision"].nda[loc] = fcio.config.blprecision
        tbl["mastercards"].nda[loc] = fcio.config.mastercards
        tbl["triggercards"].nda[loc] = fcio.config.triggercards
        tbl["adccards"].nda[loc] = fcio.config.adccards
        tbl["gps"].nda[loc] = fcio.config.gps
        ntraces = fcio.config.adcs + fcio.config.triggers
        tbl["tracemap"]._set_vector_unsafe(loc, fcio.config.tracemap[:ntraces])

        config_rb.loc += 1

        return config_rb.is_full()

    def decode_config(self, fcio: FCIO) -> lgdo.Struct:

        tbl = lgdo.Struct()

        fcio_attr_names_map = {
            "nsamples": "eventsamples",
            "nadcs": "adcs",
            "ntriggers": "triggers",
            "streamid": "streamid",
            "adcbits": "adcbits",
            "sumlength": "sumlength",
            "blprecision": "blprecision",
            "mastercards": "mastercards",
            "triggercards": "triggercards",
            "adccards": "adccards",
            "gps": "gps",
        }

        for name, fcio_attr_name in fcio_attr_names_map.items():
            if name in tbl:
                log.warning(f"{name} already in tbl. skipping...")
                continue
            value = np.int32(
                getattr(fcio.config, fcio_attr_name)
            )  # all config fields are int32
            tbl.add_field(name, lgdo.Scalar(value))
        ntraces = fcio.config.adcs + fcio.config.triggers
        tbl.add_field("tracemap", lgdo.Array(fcio.config.tracemap[:ntraces]))

        return tbl

    def get_decoded_values(self, key: int | str = None) -> dict[str, dict[str, Any]]:
        return self.decoded_values
