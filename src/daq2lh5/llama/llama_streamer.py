from __future__ import annotations

import logging
from typing import Tuple
import numpy as np

from ..data_decoder import DataDecoder
from ..data_streamer import DataStreamer
from ..raw_buffer import RawBuffer, RawBufferLibrary

from .llama_header_decoder import LLAMAHeaderDecoder, LLAMA_Channel_Configs_t
from .llama_event_decoder import LLAMAEventDecoder

log = logging.getLogger(__name__)

class LLAMAStreamer(DataStreamer):
    """
    Decode SIS3316 data acquired using llamaDAQ.
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_stream = None
        self.event_rbkd = None
        self.header_decoder = LLAMAHeaderDecoder()
        self.event_decoder = LLAMAEventDecoder()

    def get_decoder_list(self) -> list[DataDecoder]:
        dec_list = []
        dec_list.append(self.header_decoder)
        dec_list.append(self.event_decoder)
        return dec_list
    
    def open_stream(
        self,
        llama_filename: str,
        rb_lib: RawBufferLibrary = None,
        buffer_size: int = 8192,
        chunk_mode: str = "any_full",
        out_stream: str = "",
    ) -> list[RawBuffer]:
        """Initialize the LLAMA data stream.

        Refer to the documentation for
        :meth:`.data_streamer.DataStreamer.open_stream` for a description
        of the parameters.
        """

        if self.in_stream is not None:
            raise RuntimeError("tried to open stream while previous one still open")
        self.in_stream = open(llama_filename.encode("utf-8"), "rb")
        self.n_bytes_read = 0
        self.packet_id = 0

        # read header info here
        header, n_bytes_hdr = self.header_decoder.decode_header(self.in_stream)
        self.n_bytes_read += n_bytes_hdr

        self.event_decoder.set_channel_configs(self.header_decoder.get_channel_configs())

        # as far as I can tell, this happens if a user does not specify output.
        # Then I can still get a rb_lib, but that misses keys entirely, which I need since channels can have different setups.
        # So I try to hack my own here in case there is none provided.
        #if rb_lib is None:
        #    rb_lib = self.__hack_rb_lib(self.header_decoder.get_channel_configs())

        # initialize the buffers in rb_lib. Store them for fast lookup
        # Docu tells me to use initialize instead, but that does not exits (?)
        super().open_stream(
            llama_filename,
            rb_lib,
            buffer_size=buffer_size,
            chunk_mode=chunk_mode,
            out_stream=out_stream,
        )
        if rb_lib is None:
            rb_lib = self.rb_lib

        self.event_rbkd = (
            rb_lib["LLAMAEventDecoder"].get_keyed_dict()
            if "LLAMAEventDecoder" in rb_lib
            else None
        )

        if "LLAMAHeaderDecoder" in rb_lib:
            config_rb_list = rb_lib["LLAMAHeaderDecoder"]
            if len(config_rb_list) != 1:
                log.warning(
                    f"config_rb_list had length {len(config_rb_list)}, ignoring all but the first"
                )
            rb = config_rb_list[0]
        else:
            rb = RawBuffer(lgdo=header)
        rb.loc = 1  # we have filled this buffer
        return [rb]

        



    def close_stream(self) -> None:
        if self.in_stream is None:
            raise RuntimeError("tried to close an unopened stream")
        self.in_stream.close()
        self.in_stream = None

    def read_packet(self) -> bool:
        """Reads a single packet's worth of data in to the :class:`.RawBufferLibrary`.

        Returns
        -------
        still_has_data
            returns `True` while there is still data to read.
        """

        packet, fch_id = self.__read_bytes()
        if packet is None:
            return False        # EOF
        self.packet_id += 1
        self.n_bytes_read += len(packet)

        self.any_full |= self.event_decoder.decode_packet(
            packet, self.event_rbkd, self.packet_id, fch_id
        )

        return True
    
    def __read_bytes(self) -> Tuple[bytes | None, int]:
        """
        return bytes if read successful or None if EOF.
        int is the fch_id (needs to be fetched to obtain the size of the event)
        """
        if self.in_stream is None:
            raise RuntimeError("No stream open!")

        position = self.in_stream.tell()     #save position of the event header's 1st byte
        data1 = self.in_stream.read(4)       #read the first (32 bit) word of the event's header: channelID & format bits
        if len(data1) < 4:
            return None, -1         # EOF, I guess
        self.in_stream.seek(position)        #go back to 1st position of event header

        header_data_32 = np.fromstring(data1, dtype=np.uint32)
        fch_id = (header_data_32[0] >> 4) & 0x00000fff

        event_length_32 = self.header_decoder.get_channel_configs()[fch_id]["event_length"]
        event_length_8 = event_length_32 * 4

        packet = self.in_stream.read(event_length_8)
        if len(packet) < event_length_8:
            raise RuntimeError(f"Tried to read {event_length_8} bytes but got {len(packet)}")

        return packet, fch_id
    


