"""Main entry point for the procex command-line interface."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

import procex.functional as F
from procex.imgio import check_quality
from procex.imgio import read_image
from procex.imgio import write_image
from procex.imgio import write_jpeg

_app = typer.Typer(
    no_args_is_help=True,
)


class NumBits(str, Enum):
    """Number of bits per sample in the output image."""

    EIGHT = "8"
    SIXTEEN = "16"


@_app.command()
def main(  # noqa: PLR0913
    input_path: Annotated[
        Path,
        typer.Argument(),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(),
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
) -> None:
    """Resize an image."""
    image = read_image(input_path)

    if mimic:
        image = F.enhance_contrast(image, num_bits=8, histeq=True)
        write_jpeg(image, output_path, quality=95)

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


if __name__ == "__main__":
    _app()
