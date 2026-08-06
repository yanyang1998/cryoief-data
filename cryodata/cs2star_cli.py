"""Command-line interface for converting a single CS file to STAR."""

import argparse
import os
from pathlib import Path

from .cs_star_translate.cs2star import cs2star


def convert_cs_to_star(input_path, output_path=None, force=False):
    """Convert one cryoSPARC ``.cs`` file to a RELION ``.star`` file."""
    input_cs = Path(input_path).expanduser().resolve()
    if not input_cs.is_file():
        raise FileNotFoundError(f"Input CS file not found: {input_cs}")
    if input_cs.suffix.lower() != ".cs":
        raise ValueError(f"Input must have a .cs suffix: {input_cs}")

    output_star = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else input_cs.with_suffix(".star")
    )
    if output_star.suffix.lower() != ".star":
        raise ValueError(f"Output must have a .star suffix: {output_star}")
    if output_star.exists() and not force:
        raise FileExistsError(
            f"Output already exists (use force=True to replace it): {output_star}"
        )

    output_star.parent.mkdir(parents=True, exist_ok=True)
    temp_star = output_star.with_name(f".{output_star.stem}.tmp.star")
    if temp_star.exists():
        temp_star.unlink()

    try:
        result = cs2star(os.fspath(input_cs), os.fspath(temp_star))
        if result not in (None, 0) or not temp_star.exists():
            raise RuntimeError(f"Failed to convert CS file to STAR format: {input_cs}")
        os.replace(temp_star, output_star)
    finally:
        if temp_star.exists():
            temp_star.unlink()

    return os.fspath(output_star)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cryodata-cs2star",
        description="Convert one cryoSPARC .cs metadata file to RELION .star format.",
    )
    parser.add_argument("input_cs", help="Input cryoSPARC .cs file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .star path (defaults to INPUT_CS with a .star suffix)",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output STAR file")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = convert_cs_to_star(args.input_cs, args.output, force=args.force)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"cryodata-cs2star: error: {exc}\n")

    print(f"STAR: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
