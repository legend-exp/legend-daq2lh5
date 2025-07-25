"""
Base classes for decoding data into raw LGDO Tables or files
"""

from __future__ import annotations

import lgdo
import numpy as np
from lgdo import LGDO
from lgdo.lh5 import LH5Store
from lgdo.lh5 import datatype as dtypeutils


class DataDecoder:
    r"""Decodes packets from a data stream.

    Most decoders will repeatedly decode the same set of values from each
    packet.  The values that get decoded need to be described by a dict stored
    in `self.decoded_values` that helps determine how to set up the buffers and
    write them to file as :class:`~lgdo.types.lgdo.LGDO`\ s.
    :class:`~lgdo.types.table.Table`\ s are made whose columns correspond to
    the elements of `decoded_values`, and packet data gets pushed to the end of
    the table one row at a time.

    Any key-value entry in a configuration dictionary attached to an element
    of `decoded_values` is typically interpreted as an attribute to be attached
    to the corresponding LGDO. This feature can be for example exploited to
    specify HDF5 dataset settings used by
    :meth:`~lgdo.lh5.store.LH5Store.write` to write LGDOs to disk.

    For example ::

      from lgdo.compression import RadwareSigcompress

      FCEventDecoder.decoded_values = {
        "packet_id": {"dtype": "uint32", "hdf5_settings": {"compression": "gzip"}},
        # ...
        "waveform": {
          "dtype": "uint16",
          "datatype": "waveform",
          # ...
          "compression": {"values": RadwareSigcompress(codec_shift=-32768)},
          "hdf5_settings": {"t0": {"compression": "lzf", shuffle: True}},
        }
      }

    The LGDO corresponding to ``packet_id`` will have its `hdf5_settings`
    attribute set as ``{"compression": "gzip"}``, while ``waveform.values``
    will have its `compression` attribute set to
    ``RadwareSigcompress(codec_shift=-32768)``.  Before being written to disk,
    they will be compressed with the HDF5 built-in Gzip filter and with the
    :class:`~lgdo.compression.radware.RadwareSigcompress` waveform compressor.

    Examples
    --------
    See `decoded_values` attributes of
    :class:`~.fc.fc_event_decoder.FCEventDecoder` or
    :class:`~.orca.orca_digitizers.ORSIS3316WaveformDecoder`.

    Some decoders (like for file headers) do not need to push to a table, so they
    do not need `decoded_values`. Such classes should still derive from
    :class:`DataDecoder` and define how data gets formatted into LGDO's.

    Subclasses should define a method for decoding data to a buffer like
    ``decode_packet(packet, raw_buffer_list, packet_id)``.  This function
    should return the number of bytes read.

    Garbage collection writes binary data as an array of :obj:`~numpy.uint32`\ s
    to a variable-length array in the output file. If a problematic packet is
    found, call :meth:`.put_in_garbage`. User should set up an enum or bitbank
    of garbage codes to be stored along with the garbage packets.
    """

    def __init__(
        self, garbage_length: int = 256, packet_size_guess: int = 1024
    ) -> None:
        self.garbage_table = lgdo.Table(size=garbage_length)
        shape_guess = (garbage_length, packet_size_guess)
        self.garbage_table.add_field(
            "packets", lgdo.VectorOfVectors(shape_guess=shape_guess, dtype="uint8")
        )
        self.garbage_table.add_field(
            "packet_id", lgdo.Array(shape=garbage_length, dtype="uint32")
        )
        # TODO: add garbage codes enum attribute: user supplies in constructor
        # before calling super()
        self.garbage_table.add_field(
            "garbage_code", lgdo.Array(shape=garbage_length, dtype="uint32")
        )

    def get_key_lists(self) -> list[list[int | str]]:
        """Return a list of lists of keys available for this decoder.
        Each list must contain keys that can share a buffer, i.e. decoded_values
        is exactly the same (including e.g. waveform length) for all keys in the list.
        Overload with lists of keys for this decoder, e.g. ``return
        [range(n_channels)]``.  The default version works for decoders with
        single / no keys."""
        return [[None]]

    def get_decoded_values(self, key: int | str = None) -> dict:
        """Get decoded values (optionally for a given key, typically a channel).

        Notes
        -----
        Must overload for your decoder if it has key-specific decoded values.
        Must also implement ``key = None`` returns a "default"
        `decoded_values`. Otherwise, just returns :attr:`self.decoded_values`,
        which should be defined in the constructor.
        """
        if key is None:
            return self.decoded_values if hasattr(self, "decoded_values") else None

        raise NotImplementedError(
            "you need to implement key-specific get_decoded_values for",
            type(self).__name__,
        )

    def make_lgdo(self, key: int | str = None, size: int = None) -> LGDO:
        """Make an LGDO for this :class:`DataDecoder` to fill.

        This default version of this function allocates a
        :class:`~lgdo.types.table.Table` using the `decoded_values` for key. If a
        different type of LGDO object is required for this decoder, overload
        this function.

        Parameters
        ----------
        key
            used to initialize the LGDO for a particular `key` (e.g.  to have
            different trace lengths for different channels of a piece of
            hardware). Leave as ``None`` if such specialization is not
            necessary.
        size
            the size to be allocated for the LGDO, if applicable.

        Returns
        -------
        data_obj
            the newly allocated LGDO.
        """

        if not hasattr(self, "decoded_values"):
            raise AttributeError(
                type(self).__name__
                + ": no decoded_values available for setting up table"
            )

        data_obj = lgdo.Table(size=size)
        dec_vals = self.get_decoded_values(key)
        for field, fld_attrs in dec_vals.items():
            # make a copy of fld_attrs: pop off the ones we use, then keep any
            # remaining user-set attrs and store into the lgdo
            attrs = fld_attrs.copy()

            # get the dtype
            if "dtype" not in attrs:
                raise AttributeError(
                    type(self).__name__ + ": must specify dtype for", field
                )

            dtype = attrs.pop("dtype")

            # no datatype: just a "normal" array
            if "datatype" not in attrs:
                # allow to override "kind" for the dtype for lgdo
                if "kind" in attrs:
                    attrs["datatype"] = "array<1>{" + attrs.pop("kind") + "}"
                data_obj.add_field(
                    field, lgdo.Array(shape=size, dtype=dtype, attrs=attrs)
                )
                continue

            # get datatype for complex objects
            datatype = attrs.pop("datatype")

            # waveforms: must have attributes t0_units, dt, dt_units, wf_len
            if datatype == "waveform":
                t0_units = attrs.pop("t0_units")
                dt = attrs.pop("dt")
                dt_units = attrs.pop("dt_units")
                wf_len = attrs.pop("wf_len")
                settings = {
                    "compression": attrs.pop("compression", {}),
                    "hdf5_settings": attrs.pop("hdf5_settings", {}),
                }

                wf_table = lgdo.WaveformTable(
                    size=size,
                    t0=0,
                    t0_units=t0_units,
                    dt=dt,
                    dt_units=dt_units,
                    wf_len=wf_len,
                    dtype=dtype,
                    attrs=attrs,
                )

                # attach compression/hdf5_settings to sub-fields
                for el in ["values", "t0", "dt"]:
                    for settings_name in ("hdf5_settings", "compression"):
                        if el in settings[settings_name]:
                            wf_table[el].attrs[settings_name] = settings[settings_name][
                                el
                            ]

                data_obj.add_field(field, wf_table)
                continue

            # Parse datatype for remaining lgdos
            lgdotype = dtypeutils.datatype(datatype)

            # ArrayOfEqualSizedArrays
            if lgdotype is lgdo.ArrayOfEqualSizedArrays:
                length = attrs.pop("length")
                # only arrays of 1D arrays are supported at present
                dims = (1, 1)
                aoesa = lgdo.ArrayOfEqualSizedArrays(
                    shape=(size, length), dtype=dtype, dims=dims, attrs=attrs
                )
                data_obj.add_field(field, aoesa)
                continue

            # VectorOfVectors
            if lgdotype is lgdo.VectorOfVectors:
                length_guess = size
                if "length_guess" in attrs:
                    length_guess = attrs.pop("length_guess")
                vov = lgdo.VectorOfVectors(
                    shape_guess=(size, length_guess), dtype=dtype, attrs=attrs
                )
                data_obj.add_field(field, vov)
                continue

            # if we get here, got a bad datatype
            raise RuntimeError(
                type(self).__name__,
                ": do not know how to make a",
                lgdotype.__name__,
                "for",
                field,
            )

        return data_obj

    def put_in_garbage(self, packet: int, packet_id: int, code: int) -> None:
        i_row = self.garbage_table.loc
        p8 = np.frombuffer(packet, dtype="uint8")
        self.garbage_table["packets"]._set_vector_unsafe(i_row, p8)
        self.garbage_table["packet_id"].nda[i_row] = packet_id
        self.garbage_table["garbage_codes"].nda[i_row] = code
        self.garbage_table.push_row()

    def write_out_garbage(
        self, filename: str, group: str = "/", lh5_store: LH5Store = None
    ) -> None:
        n_rows = self.garbage_table.loc
        if n_rows == 0:
            return
        lgdo.lh5.write(
            self.garbage_table, "garbage", filename, group, n_rows=n_rows, append=True
        )
        self.garbage_table.clear()

    def get_max_rows_in_packet(self) -> int:
        """Returns the maximum number of rows that could be read out in a
        packet.

        1 by default, overload as necessary to avoid writing past the ends of
        buffers.
        """
        return 1
