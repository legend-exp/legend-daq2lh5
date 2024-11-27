from __future__ import annotations

import logging

from ..data_decoder import DataDecoder
from ..data_streamer import DataStreamer
from ..raw_buffer import RawBuffer, RawBufferLibrary

from .llama_header_decoder import LLAMAHeaderDecoder

log = logging.getLogger(__name__)

class LLAMAStreamer(DataStreamer):
    """
    Decode SIS3316 data acquired using llamaDAQ.
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_stream = None
        self.header_decoder = LLAMAHeaderDecoder()

    def get_decoder_list(self) -> list[DataDecoder]:
        dec_list = []
        dec_list.append(self.header_decoder)
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

        # read header info here
        header, n_bytes_hdr = self.header_decoder.decode_header(self.in_stream)
        self.n_bytes_read += n_bytes_hdr

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
        return False
