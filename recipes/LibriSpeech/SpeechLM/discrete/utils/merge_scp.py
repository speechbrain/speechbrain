"""
Script to merge sharded token files.

This script provides a command-line interface to merge sharded token files.
"""

import argparse
import pathlib as pl
import logging
from tokens import merge_sharded_tokens


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge sharded token files into a single file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory containing the sharded files",
    )

    parser.add_argument(
        "--save-name",
        type=str,
        default="tokens",
        help="Base name of the token files",
    )

    parser.add_argument(
        "--num-shards", type=int, default=1, help="Number of shards to merge"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    save_path = pl.Path(args.save_path)
    if not save_path.exists():
        raise ValueError(f"Directory not found: {save_path}")

    if args.num_shards < 1:
        raise ValueError("Number of shards must be positive")

    merge_sharded_tokens(
        save_path=args.save_path,
        save_name=args.save_name,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise