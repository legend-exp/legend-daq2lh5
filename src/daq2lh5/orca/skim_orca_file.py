import argparse
from pathlib import Path

from .orca_streamer import OrcaStreamer


def skim_orca_file() -> None:
    parser = argparse.ArgumentParser(
        prog="skim_orca_file",
        description="Convert data into LEGEND HDF5 (LH5) raw format",
    )
    parser.add_argument("in_file", type=str, help="filename of orca file to skim")
    parser.add_argument("n_packets", type=int, help="number of packets to skim")
    parser.add_argument(
        "--out-file", type=str, required=False, help="filename to write skimmed file to"
    )
    args = parser.parse_args()

    if not Path(args.in_file).is_file():
        print("file not found: {args.in_file}")  # noqa: T201
        print(  # noqa: T201
            "Usage: skim_orca_file [orca_file] [n_packets] (out_filename)"
        )

    elif args.n_packets == 0:
        print("n_packets must be a positive integer")  # noqa: T201
        print(  # noqa: T201
            "Usage: skim_orca_file [orca_file] [n_packets] (out_filename)"
        )
    else:
        if args.out_file:
            out_filename = Path(args.out_file)
        else:
            out_filename = Path(f"{args.in_file}_first{args.n_packets}")
        out_filename.parent.mkdir(parents=True, exist_ok=True)

        or_streamer = OrcaStreamer()
        or_streamer.open_stream(args.in_file)
        header_packet = or_streamer.load_packet(0)

        if out_filename.is_file():
            out_filename.unlink()

        with out_filename.open("ab") as out_file:
            # always write header and first packet (run start)
            header_packet.tofile(out_file)
            packet = or_streamer.load_packet()
            packet.tofile(out_file)
            for _ in range(args.n_packets):
                packet = or_streamer.load_packet()
                packet.tofile(out_file)
            # always write last packet (run end)
            packet = or_streamer.load_packet(1, 2)
            packet.tofile(out_file)
