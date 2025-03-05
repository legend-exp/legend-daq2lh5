from __future__ import annotations

import bz2
import gzip
import json
import logging

import numpy as np

from ..data_streamer import DataStreamer
from ..raw_buffer import RawBuffer, RawBufferLibrary
from . import orca_packet
from .orca_base import OrcaDecoder
from .orca_digitizers import (  # noqa: F401
    ORSIS3302DecoderForEnergy,
    ORSIS3316WaveformDecoder,
)
from .orca_fcio import (  # noqa: F401;
    ORFCIOConfigDecoder,
    ORFCIOEventDecoder,
    ORFCIOEventHeaderDecoder,
    ORFCIOStatusDecoder,
)
from .orca_flashcam import (  # noqa: F401;
    ORFlashCamADCWaveformDecoder,
    ORFlashCamListenerConfigDecoder,
    ORFlashCamListenerStatusDecoder,
    ORFlashCamWaveformDecoder,
)
from .orca_header_decoder import OrcaHeaderDecoder
from .orca_run_decoder import ORRunDecoderForRun  # noqa: F401

log = logging.getLogger(__name__)


class OrcaStreamer(DataStreamer):
    """Data streamer for ORCA data."""

    def __init__(self) -> None:
        super().__init__()
        self.in_stream = None
        self.packet_locs = []
        self.buffer = np.empty(1024, dtype="uint32")  # start with a 4 kB packet buffer
        self.header = None
        self.header_decoder = OrcaHeaderDecoder()
        self.decoder_id_dict = {}  # dict of data_id to decoder object
        self.rbl_id_dict = {}  # dict of RawBufferLists for each data_id
        self.missing_decoders = []

    def load_packet_header(self) -> np.ndarray | None:
        """Loads the packet header at the current read location into the buffer

        and updates internal variables.
        """
        pkt_hdr = self.buffer[:2]
        n_bytes_read = self.in_stream.readinto(pkt_hdr)  # buffer is at least 4 kB long
        self.n_bytes_read += n_bytes_read
        if n_bytes_read == 0:  # EOF
            return None
        if (n_bytes_read % 4) != 0:
            raise RuntimeError(f"got {n_bytes_read} bytes for packet header, expect 4 or 8.")
        if orca_packet.is_extended(pkt_hdr) and n_bytes_read < 8:
            raise RuntimeError(f"got {n_bytes_read} bytes for packet header, but require 8 for the extended header format.")
        if orca_packet.is_short(pkt_hdr) and n_bytes_read > 4:
            # if more than 4 bytes were read but the packet is short, we reset the file stream
            # so we can read the next uint32_t again. would not be necessary with a circular buffer
            self.in_stream.seek(n_bytes_read - 4, 1)

        # packet is valid. Can set the packet_id and log its location
        self.packet_id += 1
        filepos = self.in_stream.tell() - n_bytes_read
        if self.packet_id < len(self.packet_locs):
            if self.packet_locs[self.packet_id] != filepos:
                raise RuntimeError(
                    f"filepos for packet {self.packet_id} was {filepos} but {self.packet_locs[self.packet_id]} was expected"
                )
        else:
            if len(self.packet_locs) != self.packet_id:
                raise RuntimeError(
                    f"loaded packet {self.packet_id} after packet {len(self.packet_locs)-1}"
                )
            self.packet_locs.append(filepos)

        return pkt_hdr

    def skip_packet(self, n: int = 1) -> bool:
        """Skip a packets without loading it into the internal buffer.

        Requires loading the header. Optionally skips n packets.

        Returns
        ----------
        succeeded
            returns False if reached EOF, otherwise returns true
        """
        if self.in_stream is None:
            raise RuntimeError("self.in_stream is None")
        if not int(n) >= 0:
            raise ValueError(f"n must be a non-negative int, can't be {n}")
        n = int(n)
        while n > 0:
            pkt_hdr = self.load_packet_header()
            if pkt_hdr is None:
                return False
            self.in_stream.seek((orca_packet.get_n_words(pkt_hdr) - len(pkt_hdr)) * 4, 1)
            n -= 1
        return True

    def build_packet_locs(self, saveloc=True) -> None:
        loc = self.in_stream.tell()
        pid = self.packet_id
        if len(self.packet_locs) > 0:
            self.in_stream.seek(self.packet_locs[-1])
            self.packet_id = len(self.packet_locs) - 2
        while self.skip_packet():
            pass  # builds the rest of the packet_locs list
        if saveloc:
            self.in_stream.seek(loc)
            self.packet_id = pid

    def count_packets(self, saveloc=True) -> int:
        self.build_packet_locs(saveloc=saveloc)
        return len(self.packet_locs)

    # TODO: need to correct for endianness?
    def load_packet(
        self, index: int = None, whence: int = 0, skip_unknown_ids: bool = False
    ) -> np.ndarray | None:
        """Loads the next packet into the internal buffer.

        Returns packet as a :class:`numpy.uint32` view of the buffer (a slice),
        returns ``None`` at EOF.

        Parameters
        ----------
        index
            Optionally give an index of packet to skip to, relative to the
            "whence" location. Can be positive or negative. If out-of-range for
            the file, None will be returned.
        whence
            used when an index is supplied. Follows the file.seek() convention:
            whence = 0 (default) means index is relative to the beginning of the
            file; whence = 1 means index is relative to the current position in
            the file; whence = 2 means relative to the end of the file.

        Returns
        ----------
        packet
            a view of the internal buffer spanning the packet data (uint32
            ndarray). If you want to hold on to the packet data while you load
            more packets, you can call copy() on the view to make a copy.
        """
        if self.in_stream is None:
            raise RuntimeError("self.in_stream is None")

        if index is not None:
            if whence not in [0, 1, 2]:
                raise ValueError(f"whence can't be {whence}")
            index = int(index)
            # convert whence 1 or 2 to whence = 0
            if whence == 1:  # index is relative to current position
                index += self.packet_id - 1
            elif whence == 2:  # index is relative to end of file
                self.build_packet_locs(saveloc=False)
                index += len(self.packet_locs) - 2
            if index < 0:
                self.in_stream.seek(0)
                self.packet_id = -1
                return None
            while index >= len(self.packet_locs):
                if not self.skip_packet():
                    return None
            self.in_stream.seek(self.packet_locs[index])
            self.packet_id = index - 1

        # load packet header
        pkt_hdr = self.load_packet_header()
        if pkt_hdr is None:
            return None

        # if it's a short packet, we are done
        if orca_packet.is_short(pkt_hdr):
            return pkt_hdr

        # long packet: get length and check if we can skip it
        n_words = orca_packet.get_n_words(pkt_hdr)
        if (
            skip_unknown_ids
            and orca_packet.get_data_id(pkt_hdr, shift=False)
            not in self.decoder_id_dict
        ):
            self.in_stream.seek((n_words - len(pkt_hdr)) * 4, 1)
            return pkt_hdr

        # load into buffer, resizing as necessary
        if len(self.buffer) < n_words:
            self.buffer.resize(n_words, refcheck=False)
        n_bytes_read = self.in_stream.readinto(self.buffer[len(pkt_hdr):n_words])
        self.n_bytes_read += n_bytes_read
        if n_bytes_read != (n_words - len(pkt_hdr)) * 4:
            log.error(
                f"only got {n_bytes_read} bytes for packet read when {(n_words-1)*4} were expected. Flushing all buffers and quitting..."
            )
            return None

        # return just the packet
        return self.buffer[:n_words]

    def get_decoder_list(self) -> list[OrcaDecoder]:
        return list(self.decoder_id_dict.values())

    def set_in_stream(self, stream_name: str) -> None:
        if self.in_stream is not None:
            self.close_in_stream()
        if stream_name.endswith(".gz"):
            self.in_stream = gzip.open(stream_name.encode("utf-8"), "rb")
        elif stream_name.endswith(".bz2"):
            self.in_stream = bz2.open(stream_name.encode("utf-8"), "rb")
        else:
            self.in_stream = open(stream_name.encode("utf-8"), "rb")
        self.n_bytes_read = 0

    def close_in_stream(self) -> None:
        if self.in_stream is None:
            raise RuntimeError("tried to close an unopened stream")
        self.in_stream.close()
        self.in_stream = None

    def close_stream(self) -> None:
        self.close_in_stream()

    def is_orca_stream(stream_name: str) -> bool:  # noqa: N805
        orca = OrcaStreamer()
        orca.set_in_stream(stream_name)
        first_bytes = orca.in_stream.read(12)
        orca.close_in_stream()

        # that read should have succeeded
        if len(first_bytes) != 12:
            log.debug(f"first 12B read only returned {first_bytes}B: not orca")
            return False

        # first 14 bits should be zero
        uints = np.frombuffer(first_bytes, dtype="uint32")
        if (uints[0] & 0xFFFC0000) != 0:
            log.debug(
                f"first fourteen bits non-zero ({uints[0] & 0xFFFC0000}): not orca"
            )
            return False

        # xml header length should fit within header packet length
        pad = uints[0] * 4 - 8 - uints[1]
        if pad < 0 or pad > 3:
            log.debug(
                f"header length = {uints[1]}B doesn't fit right within header packet length = {uints[0]*4-8}B: not orca"
            )
            return False

        # last 4 chars should be '<?xm'
        if first_bytes[8:].decode() != "<?xm":
            log.debug(
                f"last 4 chars of first 12 bytes = {first_bytes[8:].decode()} != '<?xm': not orca"
            )
            return False

        # it must be an orca stream
        return True

    def hex_dump(
        self,
        stream_name: str,
        n_packets: int = np.inf,
        skip_header: bool = False,
        shift_data_id: bool = True,
        print_n_words: bool = False,
        max_words: int = np.inf,
        as_int: bool = False,
        as_short: bool = False,
    ) -> None:
        self.set_in_stream(stream_name)
        if skip_header:
            self.load_packet()
        while n_packets > 0:
            packet = self.load_packet()
            if packet is None:
                self.close_in_stream()
                return
            orca_packet.hex_dump(
                packet,
                shift_data_id=shift_data_id,
                print_n_words=print_n_words,
                max_words=max_words,
                as_int=as_int,
                as_short=as_short,
            )
            n_packets -= 1

    def open_stream(
        self,
        stream_name: str,
        rb_lib: RawBufferLibrary = None,
        buffer_size: int = 8192,
        chunk_mode: str = "any_full",
        out_stream: str = "",
    ) -> list[RawBuffer]:
        """Initialize the ORCA data stream.

        Parameters
        ----------
        stream_name
            The ORCA filename. Only file streams are currently supported.
            Socket stream reading can be added later.
        rb_lib
            library of buffers for this stream.
        buffer_size
            length of tables to be read out in :meth:`read_chunk`.
        chunk_mode : 'any_full', 'only_full', or 'single_packet'
            sets the mode use for :meth:`read_chunk`.
        out_stream
            optional name of output stream for default `rb_lib` generation.

        Returns
        -------
        header_data
            a list of length 1 containing the raw buffer holding the ORCA header.
        """

        self.set_in_stream(stream_name)
        self.packet_id = -1

        # read in the header
        packet = self.load_packet()
        if packet is None:
            raise RuntimeError(f"no orca data in file {stream_name}")
        if orca_packet.get_data_id(packet) != 0:
            raise RuntimeError(
                f"got data id {orca_packet.get_data_id(packet)} for header"
            )

        self.any_full |= self.header_decoder.decode_packet(packet, self.packet_id)
        self.header = self.header_decoder.header

        # find the names of all decoders listed in the header AND in the rb_lib (if specified)
        decoder_names = self.header.get_decoder_list()
        if rb_lib is not None and "*" not in rb_lib:
            keep_decoders = []
            for name in decoder_names:
                if name in rb_lib:
                    keep_decoders.append(name)
            decoder_names = keep_decoders
            # check that all requested decoders are present
            for name in rb_lib.keys():
                if name not in keep_decoders:
                    log.warning(
                        f"decoder {name} (requested in rb_lib) not in data description in header"
                    )

        # get a mapping of data_ids-of-interest to instantiated decoders
        id_to_dec_name_dict = self.header.get_id_to_decoder_name_dict(
            shift_data_id=False
        )
        instantiated_decoders = {"OrcaHeaderDecoder": self.header_decoder}
        for data_id in id_to_dec_name_dict.keys():
            name = id_to_dec_name_dict[data_id]
            if name not in instantiated_decoders:
                if name not in globals():
                    self.missing_decoders.append(data_id)
                    continue
                decoder = globals()[name]
                instantiated_decoders[name] = decoder(header=self.header)
            self.decoder_id_dict[data_id] = instantiated_decoders[name]

        # initialize the buffers in rb_lib. Store them for fast lookup
        super().open_stream(
            stream_name,
            rb_lib,
            buffer_size=buffer_size,
            chunk_mode=chunk_mode,
            out_stream=out_stream,
        )
        if rb_lib is None:
            rb_lib = self.rb_lib
        good_buffers = []
        for data_id in self.decoder_id_dict.keys():
            name = id_to_dec_name_dict[data_id]
            if name not in self.rb_lib:
                log.info(f"skipping data from {name}")
                continue
            self.rbl_id_dict[data_id] = self.rb_lib[name]
            good_buffers.append(name)
        # check that we have instantiated decoders for all buffers
        for key in self.rb_lib:
            if key not in good_buffers:
                log.warning(f"buffer for {key} has no decoder")
        log.debug(f"rb_lib = {self.rb_lib}")

        # return header raw buffer
        if "OrcaHeaderDecoder" in rb_lib:
            header_rb_list = rb_lib["OrcaHeaderDecoder"]
            if len(header_rb_list) != 1:
                log.warning(
                    f"header_rb_list had length {len(header_rb_list)}, ignoring all but the first"
                )
            rb = header_rb_list[0]
        else:
            rb = RawBuffer(lgdo=self.header_decoder.make_lgdo())
        rb.lgdo.value = json.dumps(self.header)
        rb.loc = 1  # we have filled this buffer
        return [rb]

    def read_packet(self) -> bool:
        """Read a packet of data.

        Data written to the `rb_lib` attribute.
        """
        # read until we get a decodeable packet
        while True:
            packet = self.load_packet(skip_unknown_ids=True)
            if packet is None:
                return False

            # look up the data id, decoder, and rbl
            data_id = orca_packet.get_data_id(packet, shift=False)
            log.debug(
                f"packet {self.packet_id}: data_id = {data_id}, decoder = {'None' if data_id not in self.decoder_id_dict else type(self.decoder_id_dict[data_id]).__name__}"
            )
            if data_id in self.missing_decoders:
                name = self.header.get_id_to_decoder_name_dict(shift_data_id=False)[
                    data_id
                ]
                log.warning(f"no implementation of {name}, packets were skipped")
                continue
            if data_id in self.rbl_id_dict:
                break

        # now decode
        decoder = self.decoder_id_dict[data_id]
        rbl = self.rbl_id_dict[data_id]
        self.any_full |= decoder.decode_packet(packet, self.packet_id, rbl)
        return True
