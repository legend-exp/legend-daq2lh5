from __future__ import annotations

import logging

import lgdo
from fcio import FCIO, Tags

from ..data_decoder import DataDecoder
from ..data_streamer import DataStreamer
from ..raw_buffer import RawBuffer, RawBufferLibrary
from .fc_config_decoder import FCConfigDecoder
from .fc_event_decoder import FCEventDecoder
from .fc_eventheader_decoder import FCEventHeaderDecoder
from .fc_fsp_decoder import FSPConfigDecoder, FSPEventDecoder, FSPStatusDecoder
from .fc_status_decoder import FCStatusDecoder

log = logging.getLogger(__name__)


class FCStreamer(DataStreamer):
    """
    Decode FlashCam data, using the ``fcio`` package to handle file access,
    and the FlashCam data decoders to save the results and write to output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fcio = FCIO()
        self.config_decoder = FCConfigDecoder()
        self.status_decoder = FCStatusDecoder()
        self.event_decoder = FCEventDecoder()
        self.eventheader_decoder = FCEventHeaderDecoder()

        self.fsp_config_decoder = FSPConfigDecoder()
        self.fsp_event_decoder = FSPEventDecoder()
        self.fsp_status_decoder = FSPStatusDecoder()

        self.event_rbkd = None
        self.eventheader_rbkd = None
        self.status_rbkd = None

        self.fsp_config_rbkd = None
        self.fsp_event_rbkd = None
        self.fsp_status_rbkd = None

        self.fcio_bytes_read = 0
        self.fcio_bytes_skipped = 0

    def get_decoder_list(self) -> list[DataDecoder]:
        dec_list = []
        dec_list.append(self.config_decoder)
        dec_list.append(self.status_decoder)
        dec_list.append(self.event_decoder)
        dec_list.append(self.eventheader_decoder)

        dec_list.append(self.fsp_config_decoder)
        dec_list.append(self.fsp_event_decoder)
        dec_list.append(self.fsp_status_decoder)

        return dec_list

    def open_stream(
        self,
        fcio_peer: str,
        rb_lib: RawBufferLibrary = None,
        buffer_size: int = 8192,
        chunk_mode: str = "any_full",
        out_stream: str = "",
    ) -> list[RawBuffer]:
        """Initialize the FlashCam data stream.

        Refer to the documentation for
        :meth:`.data_streamer.DataStreamer.open_stream` for a description
        of the parameters.

        Returns
        -------
        header_data
            a list of length 2 containing the raw buffer holding the
            :class:`~.fc_config_decoder.FCConfig` table and
            optionally the :class:`~.fsp_decoder.FSPConfig` table.
        """
        self.fcio.open(fcio_peer)  # using defaults
        self.n_bytes_read = self.fcio.read_bytes() + self.fcio.skipped_bytes()

        # read in file header (config) info, returns an lgdo.Struct
        fc_config = self.config_decoder.decode_config(self.fcio)
        config_lgdos = [fc_config]
        if self.fcio.fsp:
            config_lgdos.append(self.fsp_config_decoder.decode_config(self.fcio))

        self.status_decoder.set_fcio_stream(self.fcio)
        self.event_decoder.set_fcio_stream(self.fcio)
        self.eventheader_decoder.set_fcio_stream(self.fcio)
        self.fsp_event_decoder.set_fcio_stream(self.fcio)
        self.fsp_status_decoder.set_fcio_stream(self.fcio)

        # initialize the buffers in rb_lib. Store them for fast lookup
        super().open_stream(
            fcio_peer,
            rb_lib,
            buffer_size=buffer_size,
            chunk_mode=chunk_mode,
            out_stream=out_stream,
        )
        if rb_lib is None:
            rb_lib = self.rb_lib

        # event and status allow offer key_lists
        self.status_rbkd = (
            rb_lib["FCStatusDecoder"].get_keyed_dict()
            if "FCStatusDecoder" in rb_lib
            else None
        )

        self.event_rbkd = (
            rb_lib["FCEventDecoder"].get_keyed_dict()
            if "FCEventDecoder" in rb_lib
            else None
        )

        self.eventheader_rbkd = (
            rb_lib["FCEventHeaderDecoder"].get_keyed_dict()
            if "FCEventHeaderDecoder" in rb_lib
            else None
        )

        self.fsp_event_rbkd = (
            rb_lib["FSPEventDecoder"].get_keyed_dict()
            if "FSPEventDecoder" in rb_lib
            else None
        )

        self.fsp_status_rbkd = (
            rb_lib["FSPStatusDecoder"].get_keyed_dict()
            if "FSPStatusDecoder" in rb_lib
            else None
        )
        # set up data loop variables
        self.packet_id = 0  # for storing packet order in output tables

        rbs = []
        for config_decoder, config_lgdo in zip(
            ["FCConfigDecoder", "FSPConfigDecoder"], config_lgdos
        ):
            if config_decoder in rb_lib:
                config_rb_list = rb_lib[config_decoder]
                if len(config_rb_list) != 1:
                    log.warning(
                        f"config_rb_list for {config_decoder} had length {len(config_rb_list)}, "
                        "ignoring all but the first"
                    )
                rb = config_rb_list[0]
            else:
                rb = RawBuffer(lgdo=config_lgdo)

            # TODO: not clear if this workaround is needed, or a bug:
            # It seems like the `loc` of the RawBuffer is used as `len`
            # for individual elements in a `lgdo.Struct` while writing.
            # Search for longest and use as `loc` attr.
            if isinstance(config_lgdo, lgdo.Struct):
                max_length = max(
                    [
                        len(entry) if hasattr(entry, "__len__") else 1
                        for entry in config_lgdo.values()
                    ]
                )
                rb.loc = max_length
            rbs.append(rb)
        return rbs

    def close_stream(self) -> None:
        self.fcio.close()

    def read_packet(self) -> bool:

        if not self.fcio.get_record():
            return False  # no more data

        # The fcio stream processor (FSP) prepends records to the corresponding
        # FCIO records. The fcio library parses these separate records,
        # which requires an additional call to get_record().
        # Both records are treated as one packet, so their packet_id is shared.
        if self.fcio.tag in [Tags.FSPConfig, Tags.FSPEvent, Tags.FSPStatus]:
            if not self.fcio.get_record():
                self.n_bytes_read = self.fcio.read_bytes() + self.fcio.skipped_bytes()
                log.error(
                    f"FCIO stream ended early with a {Tags.str(self.fcio.tag)} and n_bytes_read = {self.n_bytes_read}"
                )
                return False

        self.packet_id += 1
        # records or fields in records unknown to fcio are read but not parsed,
        # and tracked in skipped_bytes
        self.n_bytes_read = self.fcio.read_bytes() + self.fcio.skipped_bytes()

        # FCIOConfigs contains header data (lengths) required to access
        # (Sparse)Event(Header) records.
        # The protocol allows updates of these settings within a datastream.
        # Concatening of FCIO streams is supported here only if the FCIOConfig
        # is the same.
        if self.fcio.tag == Tags.Config or self.fcio.tag == Tags.FSPConfig:
            log.warning(
                f"got an {Tags.str(self.fcio.tag)} after start of run? "
                f"n_bytes_read = {self.n_bytes_read}"
            )

        elif self.fcio.tag == Tags.Status:
            if self.status_rbkd is not None:
                self.any_full |= self.status_decoder.decode_packet(
                    self.fcio, self.status_rbkd, self.packet_id
                )
            if self.fcio.fsp and self.fsp_status_rbkd is not None:
                self.any_full |= self.fsp_status_decoder.decode_packet(
                    self.fcio, self.fsp_status_rbkd, self.packet_id
                )

        elif self.fcio.tag == Tags.Event or self.fcio.tag == Tags.SparseEvent:
            if self.event_rbkd is not None:
                self.any_full |= self.event_decoder.decode_packet(
                    self.fcio, self.event_rbkd, self.packet_id
                )
            if self.fcio.fsp and self.fsp_event_rbkd is not None:
                self.any_full |= self.fsp_event_decoder.decode_packet(
                    self.fcio, self.fsp_event_rbkd, self.packet_id
                )

        elif self.fcio.tag == Tags.EventHeader:
            if self.eventheader_rbkd is not None:
                self.any_full |= self.eventheader_decoder.decode_packet(
                    self.fcio, self.eventheader_rbkd, self.packet_id
                )
            if self.fcio.fsp and self.fsp_event_rbkd is not None:
                self.any_full |= self.fsp_event_decoder.decode_packet(
                    self.fcio, self.fsp_event_rbkd, self.packet_id
                )

        # FIXME: push to a buffer of skipped packets?
        else:  # unknown record
            log.warning(
                f"skipping unsupported record {Tags.str(self.fcio.tag)}. "
                f"n_bytes_read = {self.n_bytes_read}"
            )

        return True
