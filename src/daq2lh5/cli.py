"""legend-daq2lh5's command line interface utilities."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import __version__, build_raw, logging


def daq2lh5_cli():
    """daq2lh5's command line interface.

    Defines the command line interface (CLI) of the package, which exposes some
    of the most used functions to the console.  This function is added to the
    ``entry_points.console_scripts`` list and defines the ``legend-daq2lh5`` executable
    (see ``setuptools``' documentation). To learn more about the CLI, have a
    look at the help section:

    .. code-block:: console

      $ legend-daq2lh5 --help
    """

    parser = argparse.ArgumentParser(
        prog="daq2lh5", description="Convert data into LEGEND HDF5 (LH5) raw format"
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="""Print legend-daq2lh5 version and exit""",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""Increase the program verbosity""",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="""Increase the program verbosity to maximum""",
    )

    parser.add_argument(
        "in_stream",
        nargs="+",
        help="""Input stream. Can be a single file, a list of files or any
                other input type supported by the selected streamer""",
    )
    parser.add_argument(
        "--stream-type",
        "-t",
        help="""Input stream type name. Use this if the stream type cannot be
                automatically deduced by daq2lh5""",
    )
    parser.add_argument(
        "--out-spec",
        "-o",
        help="""Specification for the output stream. HDF5 or JSON file name""",
    )
    parser.add_argument(
        "--buffer_size", "-b", type=int, default=8192, help="""Set buffer size"""
    )
    parser.add_argument(
        "--max-rows",
        "-n",
        type=int,
        default=np.inf,
        help="""Maximum number of rows of data to process from the input
                file""",
    )
    parser.add_argument(
        "--overwrite", "-w", action="store_true", help="""Overwrite output files"""
    )
    parser.add_argument(
        "--kwargs",
        "-k",
        type=str,
        default="{}",
        help="""Any additional kwargs to pass to build_raw, will be parsed as a JSON string""",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.setup(logging.DEBUG)
    elif args.debug:
        logging.setup(logging.DEBUG, logging.root)
    else:
        logging.setup()

    if args.version:
        print(__version__)  # noqa: T201
        sys.exit()

    if args.kwargs:
        kwargs = json.loads(args.kwargs)

    for stream in args.in_stream:
        basename = os.path.splitext(os.path.basename(stream))[0]
        build_raw(
            stream,
            in_stream_type=args.stream_type,
            out_spec=args.out_spec,
            buffer_size=args.buffer_size,
            n_max=args.max_rows,
            overwrite=args.overwrite,
            orig_basename=basename,
            **kwargs,
        )
