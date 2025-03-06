"""
Base classes for streaming data.
"""

from __future__ import annotations

import fnmatch
import logging
from abc import ABC, abstractmethod

from .raw_buffer import RawBuffer, RawBufferLibrary, RawBufferList

log = logging.getLogger(__name__)


class DataStreamer(ABC):
    """Base clase for data streams.

    Provides a uniform interface for streaming, e.g.:

    >>> header = ds.open_stream(stream_name)
    >>> for chunk in ds: do_something(chunk)

    Also provides default management of the :class:`.RawBufferLibrary` used for
    data reading: allocation (if needed), configuration (to match the stream)
    and fill level checking.  Derived classes must define the functions
    :meth:`.get_decoder_list`, :meth:`.open_stream`, and :meth:`.read_packet`;
    see below.
    """

    def __init__(self) -> None:
        self.rb_lib = None
        self.chunk_mode = None
        self.n_bytes_read = 0
        self.any_full = False
        self.packet_id = 0

    @abstractmethod
    def open_stream(
        self,
        stream_name: str,
        rb_lib: RawBufferLibrary = None,
        buffer_size: int = 8192,
        chunk_mode: str = "any_full",
        out_stream: str = "",
    ) -> tuple[list[RawBuffer], int]:
        r"""Open and initialize a data stream.

        Open the stream, read in the header, set up the buffers.

        Call ``super().initialize([args])`` from derived class after loading
        header info to run this default version that sets up buffers in
        `rb_lib` using the stream's decoders.

        Notes
        -----
        this default version has no actual return value! You must overload this
        function, set :attr:`self.n_bytes_read` to the header packet size, and
        return the header data.

        Parameters
        ----------
        stream_name
            typically a filename or e.g. a port for streaming.
        rb_lib
            a library of buffers for readout from the data stream. `rb_lib`
            will have its LGDO's initialized during this function.
        buffer_size
            length of buffers to be read out in :meth:`read_chunk` (for buffers
            with variable length).
        chunk_mode : 'any_full', 'only_full' or 'single_packet'
            sets the mode use for :meth:`read_chunk`.
        out_stream
            optional name of output stream for default `rb_lib` generation.

        Returns
        -------
        header_data
            header_data is a list of :class:`.RawBuffer`\ 's containing any
            file header data, ready for writing to file or further processing.
            It's not a :class:`.RawBufferList` since the buffers may have a
            different format.
        """
        # call super().initialize([args]) to run this default code
        # after loading header info, then follow it with the return call.

        # store chunk mode
        self.chunk_mode = chunk_mode

        # prepare rb_lib -- its lgdo's should still be uninitialized
        if rb_lib is None:
            rb_lib = self.build_default_rb_lib(out_stream=out_stream)
        self.rb_lib = rb_lib

        # now initialize lgdo's for raw buffers
        decoders = self.get_decoder_list()
        dec_names = []
        for decoder in decoders:
            dec_name = type(decoder).__name__
            # set up wildcard decoder buffers
            if dec_name not in rb_lib:
                if "*" not in rb_lib:
                    continue  # user didn't want this decoder
                rb_lib[dec_name] = RawBufferList()
                dec_key = dec_name
                if dec_key.endswith("Decoder"):
                    dec_key = dec_key.removesuffix("Decoder")
                out_name = rb_lib["*"][0].out_name.format(name=dec_key)
                out_stream = rb_lib["*"][0].out_stream.format(name=dec_key)
                proc_spec = rb_lib["*"][0].proc_spec
                key_lists = decoder.get_key_lists()
                for ii, key_list in enumerate(key_lists):
                    this_name = out_name
                    if len(key_lists) > 1:
                        if len(key_list) == 1:
                            this_name = f"{out_name}_{key_list[0]}"
                        else:
                            this_name = f"{out_name}_{ii}"
                    rb = RawBuffer(
                        key_list=key_list,
                        out_stream=out_stream,
                        out_name=this_name,
                        proc_spec=proc_spec,
                    )
                    rb_lib[dec_name].append(rb)

            # dec_name is in rb_lib: store the name, and initialize its buffer lgdos
            dec_names.append(dec_name)

            # Parse wildcard keys in RawBuffers and replace with known keys of the decoder.
            dec_key_list = sum(decoder.get_key_lists(), [])
            dec_key_type = type(dec_key_list[0]) # requires consistent types
            dec_key_list = set(map(str,dec_key_list))

            log.debug(f"{dec_name} offers keys {dec_key_list}")

            # track keys which are already used
            matched_keys = set()
            only_wildcard_rb = None
            wildcard_rbs = []
            # find wildcard key buffers
            for rb in rb_lib[dec_name]:
                for key in rb.key_list:
                    if not isinstance(key, str):
                        matched_keys.add(key)
                        continue
                    if key == "*":
                        if only_wildcard_rb is None:
                            only_wildcard_rb = rb
                        else:
                            raise KeyError(f"Only one '*' wildcard key allowed for decoder {dec_name}")

                    elif "*" in key:
                        wildcard_rbs.append(rb)
                    else:
                        matched_keys.add(key)
            # append pure wildcard, so it matches last
            if only_wildcard_rb is not None:
                wildcard_rbs.append(only_wildcard_rb)

            # remove already matched keys
            dec_key_list = dec_key_list.symmetric_difference(matched_keys)

            for rb in wildcard_rbs:
                matched_keys = set()
                for key in rb.key_list:
                    # find matching keys in the decoder list
                    matches = set(fnmatch.filter(dec_key_list, key))
                    dec_key_list = dec_key_list.symmetric_difference(matches)

                    log.debug(f"{dec_name} {key} matched keys: {matches}")
                    log.debug(f"{dec_name} remaining keys: {dec_key_list}")
                    matched_keys |= matches

                # Construct the new key_list for the RawBuffer
                # Expect anything that can be cast to int wants to be cast
                rb.key_list = []
                for key in matched_keys:
                    if key == 'None':
                        rb.key_list.append(None)
                    try:
                        new_key = int(key)
                        rb.key_list.append(new_key)
                    except ValueError:
                        rb.key_list.append(key)


                if len(rb.key_list) == 0:
                    log.warning(f"no matched keys for key_list {rb.key_list} in {dec_name}.{rb.out_name}")
                log.debug(f"{dec_name}:{rb.out_stream}/{rb.out_name} matched wildcards to {rb.key_list}")

            keyed_name_rbs = []
            ii = 0
            while ii < len(rb_lib[dec_name]):
                if "{key" in rb_lib[dec_name][ii].out_name:
                    keyed_name_rbs.append(rb_lib[dec_name].pop(ii))
                else:
                    ii += 1
            for rb in keyed_name_rbs:
                for key in rb.key_list:
                    # keys can be strs or ints; try as-is, but can get a
                    # ValueError e.g. when using a wildcard with int keys. In
                    # that case, switch to the other type and try again
                    try:
                        expanded_name = rb.out_name.format(key=key)
                    except ValueError:
                        if isinstance(key, str):
                            key = int(key)
                        else:
                            key = str(key)
                        expanded_name = rb.out_name.format(key=key)
                    new_rb = RawBuffer(
                        key_list=[key],
                        out_stream=rb.out_stream,
                        out_name=expanded_name,
                        proc_spec=rb.proc_spec,
                    )
                    rb_lib[dec_name].append(new_rb)

            for rb in rb_lib[dec_name]:
                # use the first available key
                key = rb.key_list[0] if len(rb.key_list) > 0 else None
                rb.lgdo = decoder.make_lgdo(key=key, size=buffer_size)
                rb.fill_safety = decoder.get_max_rows_in_packet()
                if buffer_size < rb.fill_safety:
                    raise ValueError(
                        f"{dec_name} requires a buffer of at least length "
                        f"{rb.fill_safety} but buffer size is only {buffer_size}"
                    )

        # make sure there were no entries in rb_lib that weren't among the
        # decoders. If so, just emit a warning and continue.
        if "*" in rb_lib:
            rb_lib.pop("*")
        for dec_name in rb_lib.keys():
            if dec_name not in dec_names:
                log.warning(f"no decoder named '{dec_name}' requested by rb_lib")

    @abstractmethod
    def close_stream(self) -> None:
        """Close this data stream.

        .. note::
            Needs to be overloaded.
        """
        pass

    @abstractmethod
    def read_packet(self) -> bool:
        """Reads a single packet's worth of data in to the :class:`.RawBufferLibrary`.

        Needs to be overloaded. Gets called by :meth:`.read_chunk` Needs to
        update :attr:`self.any_full` if any buffers would possibly over-fill on
        the next read. Needs to update :attr:`self.n_bytes_read` too.

        Returns
        -------
        still_has_data
            returns `True` while there is still data to read.
        """
        return True

    def read_chunk(
        self,
        chunk_mode_override: str = None,
        rp_max: int = 1000000,
        clear_full_buffers: bool = True,
    ) -> tuple[list[RawBuffer], int]:
        """Reads a chunk of data into raw buffers.

        Reads packets until at least one buffer is too full to perform another
        read. Default version just calls :meth:`.read_packet` over and over.
        Overload as necessary.

        Notes
        -----
        user is responsible for resetting / clearing the raw buffers prior to
        calling :meth:`.read_chunk` again.

        Parameters
        ----------
        chunk_mode_override : 'any_full', 'only_full' or 'single_packet'
            - ``None`` : do not override self.chunk_mode
            - ``any_full`` : returns all raw buffers with data as soon as any one
              buffer gets full
            - ``only_full`` : returns only those raw buffers that became full (or
              nearly full) during the read. This minimizes the number of write calls.
            - ``single_packet`` : returns all raw buffers with data after a single
              read is performed. This is useful for streaming data out as soon
              as it is read in (e.g. for diagnostics or in-line analysis).
        rp_max
            maximum number of packets to read before returning anyway, even if
            one of the other conditions is not met.
        clear_full_buffers
            automatically clear any buffers that report themselves as being
            full prior to reading the chunk. Set to `False` if clearing
            manually for a minor speed-up.

        Returns
        -------
        chunk_list : list of RawBuffers, int
            chunk_list is the list of RawBuffers with data ready for writing to
            file or further processing. The list contains all buffers with data
            or just all full buffers depending on the flag full_only.  Note
            chunk_list is not a RawBufferList since the RawBuffers inside may
            not all have the same structure
        """

        if clear_full_buffers:
            self.rb_lib.clear_full()
        self.any_full = False

        chunk_mode = (
            self.chunk_mode if chunk_mode_override is None else chunk_mode_override
        )

        read_one_packet = chunk_mode == "single_packet"
        only_full = chunk_mode == "only_full"

        n_packets = 0
        still_has_data = True
        while True:
            still_has_data = self.read_packet()
            if not still_has_data:
                break
            n_packets += 1
            if read_one_packet or n_packets > rp_max:
                break
            if self.any_full:
                break

        # send back all rb's with data if we finished reading
        if not still_has_data:
            only_full = False

        list_of_rbs = []
        for rb_list in self.rb_lib.values():
            for rb in rb_list:
                if not only_full:  # any_full or read_one_packet
                    if rb.loc > 0:
                        list_of_rbs.append(rb)
                elif rb.is_full():
                    list_of_rbs.append(rb)
        if not still_has_data:
            log.debug(f"decoding complete. flushing {len(list_of_rbs)} buffers")
        return list_of_rbs

    @abstractmethod
    def get_decoder_list(self) -> list:
        """Returns a list of decoder objects for this data stream.

        Notes
        -----
        Needs to be overloaded. Gets called during :meth:`.open_stream`.
        """
        return []

    def build_default_rb_lib(self, out_stream: str = "") -> RawBufferLibrary:
        """Build the most basic :class:`~.RawBufferLibrary` that will work for
        this stream.

        A :class:`.RawBufferList` containing a single :class:`.RawBuffer` is
        built for each decoder name returned by :meth:`.get_decoder_list`. Each
        buffer's `out_name` is set to the decoder name. The LGDO's do not get
        initialized.
        """
        rb_lib = RawBufferLibrary()
        decoders = self.get_decoder_list()
        log.debug(f"Default rb_lib knows about: {decoders}")
        if len(decoders) == 0:
            log.warning(
                f"no decoders returned by get_decoder_list() for {type(self).__name__}"
            )
            return rb_lib
        for decoder in decoders:
            dec_name = type(decoder).__name__
            dec_key = dec_name
            if dec_key.endswith("Decoder"):
                dec_key = dec_key.removesuffix("Decoder")
            key_lists = decoder.get_key_lists()
            log.debug(f"{dec_key} supports keys {key_lists}")
            for ii, key_list in enumerate(key_lists):
                this_name = dec_key
                if len(key_lists) > 1:
                    if len(key_list) == 1:
                        this_name = f"{dec_key}_{key_list[0]}"
                    else:
                        this_name = f"{dec_key}_{ii}"  # this can cause a name clash e.g. for [[1],[2,3]] ...
                rb = RawBuffer(
                    key_list=key_list, out_stream=out_stream, out_name=this_name
                )
                if dec_name not in rb_lib:
                    rb_lib[dec_name] = RawBufferList()
                rb_lib[dec_name].append(rb)
        return rb_lib
