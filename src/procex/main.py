"""Main entry point for the procex command-line interface."""

from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

import procex.functional as F
from procex.imgio import check_quality
from procex.imgio import read_image
from procex.imgio import write_image
from procex.imgio import write_jpeg

_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
)


class NumBits(str, Enum):
    """Number of bits per sample in the output image."""

    EIGHT = "8"
    SIXTEEN = "16"


@_app.command()
def process_images(  # noqa: PLR0913
    input: Annotated[  # noqa: A002
        Path,
        typer.Argument(
            ...,
            help=(
                "Path to the input image. If a text file is given, process the image"
                " paths from the file. If a directory is given, process all files in"
                " the directory."
            ),
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(
            ...,
            help=(
                "Path to the output image. If a text file is given, the output paths"
                " must be specified in the file. If a directory is given, write the"
                " output images to the directory."
            ),
        ),
    ],
    size: Annotated[
        int | None,
        typer.Option(
            ...,
            help="Size of the smaller side of the output image.",
        ),
    ] = None,
    num_bits: Annotated[
        NumBits,
        typer.Option(
            ...,
            help="Number of bits per sample in the output image.",
        ),
    ] = NumBits.EIGHT,
    jpeg_quality: Annotated[
        int,
        typer.Option(
            ...,
            help="Compression quality for output JPEG images.",
            callback=check_quality,
        ),
    ] = 95,
    percentiles: Annotated[
        tuple[float, float],
        typer.Option(
            ...,
            help="Lower and upper percentiles to clip the image intensity.",
        ),
    ] = (0, 100),
    values: Annotated[
        tuple[float, float] | None,
        typer.Option(
            ...,
            help="Lower and upper values to clip the image intensity.",
        ),
    ] = None,
    format: Annotated[  # noqa: A002
        str | None,
        typer.Option(
            ...,
            help="Output image format. Only used when output is a directory.",
        ),
    ] = None,
    *,
    histeq: Annotated[
        bool,
        typer.Option(
            ...,
            help=(
                "Whether to perform histogram equalization instead of intensity range"
                " stretching."
            ),
        ),
    ] = False,
    mimic: Annotated[
        bool,
        typer.Option(
            ...,
            help="Ignore all other options and process as in MIMIC-CXR-JPG.",
        ),
    ] = False,
    parallel: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to process images in parallel.",
        ),
    ] = False,
) -> None:
    """Preprocess a medical image."""
    input_paths = _get_input_paths(input)
    output_paths = _get_output_paths(output, input_paths, format=format)

    _process = partial(
        _process_image,
        size=size,
        num_bits=num_bits,
        jpeg_quality=jpeg_quality,
        percentiles=percentiles,
        values=values,
        histeq=histeq,
        mimic=mimic,
    )

    if parallel:
        process_map(_process, input_paths, output_paths, chunksize=1)
        return

    progress = tqdm(list(zip(input_paths, output_paths, strict=True)))
    for input_path, output_path in progress:
        _process(input_path, output_path)


def _process_image(  # noqa: PLR0913
    input_path: Path,
    output_path: Path,
    size: int | None,
    num_bits: NumBits,
    jpeg_quality: int,
    percentiles: tuple[float, float],
    values: tuple[float, float] | None,
    *,
    histeq: bool,
    mimic: bool,
) -> None:
    image = read_image(input_path)

    if mimic:
        image = F.enhance_contrast(image, num_bits=8, histeq=True)
        if output_path.suffix not in {".jpg", ".jpeg"}:
            output_path = output_path.with_suffix(".jpg")
        write_jpeg(image, output_path, quality=95)
        return

    if size is not None:
        image = F.resize(image, size)

    image = F.enhance_contrast(
        image,
        num_bits=int(num_bits),
        percentiles=percentiles,
        values=values,
        histeq=histeq,
    )

    match output_path.suffix:
        case ".jpg" | ".jpeg":
            write_jpeg(image, output_path, jpeg_quality)
        case _:
            write_image(image, output_path)


def _get_input_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        paths = sorted(input_path.iterdir())
    elif input_path.suffix == ".txt":
        paths = [Path(p) for p in input_path.read_text().strip().splitlines()]
    elif input_path.is_file():
        paths = [input_path]
    else:
        message = f"Invalid input path: {input_path}"
        raise ValueError(message)
    return paths


def _get_output_paths(
    output_path: Path,
    input_paths: list[Path],
    format: str | None,  # noqa: A002
) -> list[Path]:
    if output_path.suffix == ".txt":
        paths = [Path(p) for p in output_path.read_text().strip().splitlines()]
        if len(paths) != len(input_paths):
            message = (
                f"Number of input images ({len(input_paths)}) does not match the number"
                f" of output paths ({len(paths)})"
            )
            raise ValueError(message)
    elif output_path.is_file():
        paths = [output_path]
    elif output_path.is_dir():
        paths = [output_path / p.name for p in input_paths]
        if format is not None:
            paths = [p.with_suffix(f".{format.lstrip(".")}") for p in paths]
    else:
        message = f"Invalid output path: {output_path}"
        raise ValueError(message)
    return paths


if __name__ == "__main__":
    _app()
