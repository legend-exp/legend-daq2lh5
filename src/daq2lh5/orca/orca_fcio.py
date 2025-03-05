import copy
import gc
import logging
from typing import Any
import copy

import numpy as np

from fcio import Limits, Tags, FCIO

from daq2lh5.fc.fc_config_decoder import FCConfigDecoder
from daq2lh5.fc.fc_event_decoder import FCEventDecoder
from daq2lh5.fc.fc_fsp_decoder import FSPConfigDecoder, FSPEventDecoder, FSPStatusDecoder
from daq2lh5.fc.fc_eventheader_decoder import FCEventHeaderDecoder, get_key, get_fcid, get_card_address, get_card_input

from daq2lh5.fc.fc_status_decoder import FCStatusDecoder
from daq2lh5.fc.fc_status_decoder import get_key as get_status_key
from daq2lh5.fc.fc_status_decoder import get_fcid as get_status_fcid

from ..raw_buffer import RawBufferLibrary, RawBufferList, RawBuffer
from .orca_base import OrcaDecoder
from .orca_header import OrcaHeader
from .orca_packet import OrcaPacket, is_extended

import lgdo

log = logging.getLogger(__name__)


# using multiple decoders for the FCIO stream
# requires storing the FCIO object (and it's state) globally
fcio_stream_library = dict()

def get_fcio_stream(streamid):
    if streamid in fcio_stream_library:
        return fcio_stream_library[streamid]
    else:
        fcio_stream_library[streamid] = FCIO()
        return fcio_stream_library[streamid]

def extract_header_information(header: OrcaHeader):

    fc_hdr_info = {
      "key_list" : {}, # access by fcid
      "n_adc" : {},
      "adc_card_layout" : {}, # [fcid][key] = (crate, card, cardaddress)
      "wf_len" : {}, # [fcid] = wf_len
      "fsp_enabled": {}, # [fcid]
      "n_card" : {},
    }

    fc_card_info_dict = header.get_object_info([
        "ORFlashCamGlobalTriggerModel",
        "ORFlashCamTriggerModel",
        "ORFlashCamADCModel",
        "ORFlashCamADCStdModel",
        ])


    log.debug(f"CardInfoDict")
    for crate in fc_card_info_dict:
        for card in fc_card_info_dict[crate]:
            log.debug(f"crate {crate} card {card} {fc_card_info_dict[crate][card]}")

    fc_listener_info_list = header.get_readout_info("ORFlashCamListenerModel")
    for fc_listener_info in fc_listener_info_list:
        fcid = fc_listener_info["uniqueID"] # it should be called listener_id
        if fcid == 0:
            raise ValueError("got fcid=0 unexpectedly!")
        fc_hdr_info["fsp_enabled"][fcid] = header.get_auxhw_info("ORFlashCamListenerModel", fcid)["fspEnabled"]

        # get FC card object info from header to use below
        # gives access like fc_info[crate][card]

        fc_hdr_info["wf_len"][fcid] = header.get_auxhw_info("ORFlashCamListenerModel", fcid)["eventSamples"]
        fc_hdr_info["n_adc"][fcid] = 0
        fc_hdr_info["n_card"][fcid] = 0
        fc_hdr_info["key_list"][fcid] = []
        fc_hdr_info["adc_card_layout"][fcid] = {}
        for child in fc_listener_info["children"]:

            crate = child["crate"]
            card = child["station"]
            card_address = fc_card_info_dict[crate][card]["CardAddress"]
            fc_hdr_info["adc_card_layout"][fcid][card_address] = (crate, card, card_address)
            fc_hdr_info["n_card"][fcid] += 1

            log.debug(f"fcid {fcid} has crate {crate} card {card} {fc_card_info_dict[crate][card]['Class Name']}")

            if crate not in fc_card_info_dict:
                raise RuntimeError(f"no crate {crate} in fc_card_info_dict")
            if card not in fc_card_info_dict[crate]:
                raise RuntimeError(f"no card {card} in fc_card_info_dict[{crate}]")

            for fc_input in range(len(fc_card_info_dict[crate][card]["Enabled"])):
                if not fc_card_info_dict[crate][card]["Enabled"][fc_input]:
                    continue

                fc_hdr_info["n_adc"][fcid] += 1
                key = get_key(fcid, card_address, fc_input)

                if key in fc_hdr_info["key_list"][fcid]:
                    log.warning(f"key {key} already in key_list...")
                else:
                    fc_hdr_info["key_list"][fcid].append(key)


    return fc_hdr_info



class ORFCIOConfigDecoder(OrcaDecoder):
    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:

        self.decoder = FCConfigDecoder()
        self.decoded_values = {}

        super().__init__(header=header, **kwargs)

    def set_header(self, header: OrcaHeader) -> None:
        self.header = header
        self.decoded_values = copy.deepcopy(self.decoder.get_decoded_values())

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferList
    ) -> bool:

        fcio_stream = get_fcio_stream(packet[2])
        if fcio_stream.is_open():
            log.warning(f"FCIO stream with stream id {packet[2]} already opened. Continue with updated FCIOConfig.")
            fcio_stream.set_mem_field(memoryview(packet[3:]))
        else:
            fcio_stream.open(memoryview(packet[3:]))

        if fcio_stream.config.streamid != packet[2]:
            log.warning(f"The expected stream id {packet[2]} does not match the contained stream id {fcio_stream.config.streamid}")

        any_full = self.decoder.decode_packet(fcio_stream, rbl[0], packet_id)

        return bool(any_full)



class ORFCIOStatusDecoder(OrcaDecoder):
    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:

        self.decoder = FCStatusDecoder()
        self.decoded_values = {}
        self.key_list = {
          'status' : [],
          'fspstatus' : []
        }
        self.max_rows_in_packet = 0
        super().__init__(header=header, **kwargs)

    def set_header(self, header: OrcaHeader) -> None:
        """Setter for headers. Overload to set card parameters, etc."""
        self.header = header
        self.fc_hdr_info = extract_header_information(header)
        self.decoded_values = copy.deepcopy(self.decoder.get_decoded_values())

        for fcid in self.fc_hdr_info['n_card']:
            self.key_list['status'] = [get_status_key(fcid, 0)] # we pretent we have a master, if it's not there the
            self.key_list['status'] += [get_status_key(fcid, 0x2000 + i) for i in range(self.fc_hdr_info['n_card'][fcid])]
            self.decoded_values[fcid] = copy.deepcopy(self.decoder.get_decoded_values())
            if self.fc_hdr_info["fsp_enabled"][fcid]:
                self.key_list['fspstatus'].append(get_key(fcid,0,0))
                self.fsp_decoder = FSPStatusDecoder()
        self.max_rows_in_packet = max(self.fc_hdr_info["n_card"].values()) + 1

    def get_key_lists(self) -> list[list[int|str]]:
        return list(self.key_list.values())

    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        if key is None:
            dec_vals_list = list(self.decoded_values.values())
            if len(dec_vals_list) > 0:
                return dec_vals_list[0]
            raise RuntimeError("decoded_values not built")
        fcid = get_fcid(key)
        if fcid * 1e6 == key and self.fsp_decoder is not None:
            return self.fsp_decoder.get_decoded_values()
        fcid = get_status_fcid(key)
        if fcid in self.decoded_values:
            return self.decoded_values[fcid]
        raise KeyError(f"no decoded values for key {key} (fcid {fcid})")

    def get_max_rows_in_packet(self) -> int:
        return self.max_rows_in_packet

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferList
    ) -> bool:
        status_rbkd = rbl.get_keyed_dict()

        fcio_stream = get_fcio_stream(packet[2])
        fcio_stream.set_mem_field(memoryview(packet[3:]))

        self.decoder.set_fcio_stream(fcio_stream)
        self.fsp_decoder.set_fcio_stream(fcio_stream)

        any_full = False
        while fcio_stream.get_record():
            if fcio_stream.tag == Tags.Status:
                any_full |= self.decoder.decode_packet(fcio_stream, status_rbkd, packet_id)

        return bool(any_full)



class ORFCIOEventHeaderDecoder(OrcaDecoder):
    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:

        self.decoder = FCEventHeaderDecoder()
        self.fsp_decoder = None
        self.decoded_values = {}
        self.key_list = []

        super().__init__(header=header, **kwargs)

    def set_header(self, header: OrcaHeader) -> None:
        """Setter for headers. Overload to set card parameters, etc."""
        self.header = header

        self.fc_hdr_info = extract_header_information(header)

        key_list = self.fc_hdr_info['key_list']
        for fcid in key_list:
            self.key_list.append(get_key(fcid,0,0))
            self.decoded_values[fcid] = copy.deepcopy(self.decoder.get_decoded_values())
            if self.fc_hdr_info["fsp_enabled"][fcid]:
                self.fsp_decoder = FSPEventDecoder()
                self.decoded_values[fcid] |= copy.deepcopy(self.fsp_decoder.get_decoded_values())

        self.max_rows_in_packet = max(self.fc_hdr_info["n_adc"].values())

    def get_key_lists(self) -> list[list[int]]:
        return [self.key_list]

    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        if key is None:
            dec_vals_list = list(self.decoded_values.values())
            if len(dec_vals_list) > 0:
                return dec_vals_list[0]
            raise RuntimeError("decoded_values not built")
        fcid = get_fcid(key)
        # if get_card_address(key) == 0 and get_card_input(key) == 0 and self.fsp_decoder is not None:
        #     return self.fsp_decoder.get_decoded_values()
        if fcid in self.decoded_values:
            return self.decoded_values[fcid]
            # return self.decoder.get_decoded_values()
        raise KeyError(f"no decoded values for key {key} (fcid {fcid})")

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferList
    ) -> bool:
        evthdr_rbkd = rbl.get_keyed_dict()
        fcio_stream = get_fcio_stream(packet[2])
        fcio_stream.set_mem_field(memoryview(packet[3:]))
        self.decoder.set_fcio_stream(fcio_stream)
        if self.fsp_decoder is not None:
            self.fsp_decoder.set_fcio_stream(fcio_stream)

        any_full = False
        while fcio_stream.get_record():
            if fcio_stream.tag == Tags.EventHeader:
                any_full |= self.decoder.decode_packet(fcio_stream, evthdr_rbkd, packet_id)
                if self.fsp_decoder is not None:
                    any_full |= self.fsp_decoder.decode_packet(fcio_stream, evthdr_rbkd, packet_id)

        return bool(any_full)



class ORFCIOEventDecoder(OrcaDecoder):
    """Decoder for FlashCam FCIO stream data written by ORCA."""

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:
        self.decoder = FCEventDecoder()
        self.fsp_decoder = None

        self.key_list = {
          'event' : [],
          'fspevent' : []
        }
        self.decoded_values = {}
        self.max_rows_in_packet = 0

        # self.skipped_channels = {}
        super().__init__(header=header, **kwargs)

    def set_header(self, header: OrcaHeader) -> None:
        """Setter for headers. Overload to set card parameters, etc."""
        self.header = header
        # self.key_list = copy.deepcopy(self.decoder.get_key_lists())
        self.fc_hdr_info = extract_header_information(header)
        key_list = self.fc_hdr_info['key_list']
        for fcid in key_list:
            self.key_list['event'] += key_list[fcid]
            self.decoded_values[fcid] = copy.deepcopy(self.decoder.get_decoded_values())
            self.decoded_values[fcid]["waveform"]["wf_len"] = self.fc_hdr_info["wf_len"][fcid]
            if self.fc_hdr_info["fsp_enabled"][fcid]:
                self.key_list['fspevent'].append(get_key(fcid,0,0))
                self.fsp_decoder = FSPEventDecoder()
        self.max_rows_in_packet = max(self.fc_hdr_info["n_adc"].values())

    def get_key_lists(self) -> list[list[int]]:
        return list(self.key_list.values())

    def get_max_rows_in_packet(self) -> int:
        return self.max_rows_in_packet

    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        if key is None:
            dec_vals_list = list(self.decoded_values.values())
            if len(dec_vals_list) > 0:
                return dec_vals_list[0]
            raise RuntimeError("decoded_values not built")
        fcid = get_fcid(key)
        if get_card_address(key) == get_card_input(key) == 0 and self.fsp_decoder is not None:
            return self.fsp_decoder.get_decoded_values()
        if fcid in self.decoded_values:
            return self.decoded_values[fcid]
        raise KeyError(f"no decoded values for key {key} (fcid {fcid})")

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferList
    ) -> bool:
        """Decode the ORCA FlashCam ADC packet."""
        evt_rbkd = rbl.get_keyed_dict()

        fcio_stream = get_fcio_stream(packet[2])
        fcio_stream.set_mem_field(memoryview(packet[3:]))

        self.decoder.set_fcio_stream(fcio_stream)
        self.fsp_decoder.set_fcio_stream(fcio_stream)

        any_full = False
        while fcio_stream.get_record():
            if fcio_stream.tag == Tags.Event or fcio_stream.tag == Tags.SparseEvent:
                any_full |= self.decoder.decode_packet(fcio_stream, evt_rbkd, packet_id)
                if self.fsp_decoder is not None:
                    any_full |= self.fsp_decoder.decode_packet(fcio_stream, evt_rbkd, packet_id)

        return bool(any_full)
