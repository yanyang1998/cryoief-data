"""Prepare sorted CryoSPARC particle metadata for downstream tools."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .cs_star_translate.cs2star import cs2star
from .data_preprocess.mrc_preprocess import (
    load_original_csdata_from_cryosparc_dir,
    sort_csdata,
)


@dataclass(frozen=True)
class PreparedParticleMetadata:
    cs_path: str
    star_path: Optional[str]
    particle_count: int
    source_paths: tuple[str, ...]


def prepare_particle_metadata(input_dir, output_dir=None, write_star=True, force=False):
    """Build sorted ``new_particles.cs`` and optionally ``new_particles.star``."""
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"CryoSPARC input directory not found: {input_path}")

    output_path = Path(output_dir).expanduser().resolve() if output_dir else input_path
    cs_path = output_path / "new_particles.cs"
    star_path = output_path / "new_particles.star"
    targets = [cs_path] + ([star_path] if write_star else [])
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        joined = ", ".join(os.fspath(path) for path in existing)
        raise FileExistsError(f"Output already exists (use force=True to replace it): {joined}")

    cs_data, source_paths = load_original_csdata_from_cryosparc_dir(input_path)
    fields = set(cs_data.fields())
    missing = sorted({"blob/path", "blob/idx"} - fields)
    if missing:
        raise ValueError(f"Particle metadata is missing required fields: {', '.join(missing)}")
    if len(cs_data) == 0:
        raise ValueError("Particle metadata contains no matching particles")

    output_path.mkdir(parents=True, exist_ok=True)
    temp_cs = output_path / ".new_particles.tmp.cs"
    temp_star = output_path / ".new_particles.tmp.star"
    for path in (temp_cs, temp_star):
        if path.exists():
            path.unlink()

    try:
        new_cs_data, _, _ = sort_csdata(cs_data, os.fspath(temp_cs))
        if write_star:
            result = cs2star(os.fspath(temp_cs), os.fspath(temp_star))
            if result not in (None, 0) or not temp_star.exists():
                raise RuntimeError("Failed to convert new_particles.cs to STAR format")
        os.replace(temp_cs, cs_path)
        if write_star:
            os.replace(temp_star, star_path)
    finally:
        for path in (temp_cs, temp_star):
            if path.exists():
                path.unlink()

    return PreparedParticleMetadata(
        cs_path=os.fspath(cs_path),
        star_path=os.fspath(star_path) if write_star else None,
        particle_count=len(new_cs_data),
        source_paths=tuple(os.fspath(path) for path in source_paths),
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cryodata-prepare-cs",
        description="Build sorted new_particles.cs and new_particles.star metadata files.",
    )
    parser.add_argument("input_dir", help="CryoSPARC job or export directory")
    parser.add_argument("--output-dir", help="Output directory (defaults to INPUT_DIR)")
    parser.add_argument("--no-star", action="store_true", help="Do not generate new_particles.star")
    parser.add_argument("--force", action="store_true", help="Replace existing requested outputs")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = prepare_particle_metadata(
            args.input_dir,
            output_dir=args.output_dir,
            write_star=not args.no_star,
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"cryodata-prepare-cs: error: {exc}\n")

    print("Sources:")
    for source in result.source_paths:
        print(f"  {source}")
    print(f"Particles: {result.particle_count}")
    print(f"CS: {result.cs_path}")
    if result.star_path is not None:
        print(f"STAR: {result.star_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
