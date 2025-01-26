"""Input/output utilities for image processing."""

import enum
from collections.abc import Callable
from pathlib import Path

import SimpleITK as sitk

from .functional import rgb2gray
from .functional import squeeze as squeeze_image
from .type_definitions import TypePath


@enum.unique
class ItkImageIo(str, enum.Enum):
    """Enumeration of ITK image IO classes."""

    JPEG = "JPEGImageIO"
    JPEG_2000 = "JPEG2000ImageIO"
    PNG = "PNGImageIO"
    TIFF = "TIFFImageIO"


def read_image(
    path: TypePath,
    *,
    squeeze: bool = True,
    grayscale: bool = True,
) -> sitk.Image:
    """Read an image from a file.

    Args:
        path: The path to the image file.
        squeeze: Whether to remove singleton dimensions from the image.
        grayscale: Whether to convert the image to single-channel grayscale.

    Returns:
        The image read from the file
    """
    image = sitk.ReadImage(str(path))
    if grayscale:
        image = rgb2gray(image)
    if squeeze:
        image = squeeze_image(image)
    return image


def write_jpeg_2000(
    image: sitk.Image,
    path: TypePath,
) -> None:
    _check_suffix(path, ".jp2")
    write_image(image, path)


def write_image(
    image: sitk.Image,
    path: TypePath,
) -> None:
    sitk.WriteImage(image, str(path))


def write_tiff(
    image: sitk.Image,
    path: TypePath,
) -> None:
    _check_suffix(path, (".tif", ".tiff"))
    write_image(image, path)


def check_uint8(func: Callable) -> Callable:
    def wrapper(
        image: sitk.Image,
        path: TypePath,
        *args,
        **kwargs,
    ) -> None:
        if image.GetPixelID() != sitk.sitkUInt8:
            msg = (
                f'Expected image "{path}" to have pixel type "8-bit unsigned integer",'
                f' but got "{image.GetPixelIDTypeAsString()}"'
            )
            raise ValueError(msg)
        return func(image, path, *args, **kwargs)

    return wrapper


@check_uint8
def write_jpeg(
    image: sitk.Image,
    path: TypePath,
    quality: int = 95,  # default in ITK: https://github.com/InsightSoftwareConsortium/ITK/blob/15af3aed65693811448c9af22ce9d09ff9f3000a/Modules/IO/JPEG/src/itkJPEGImageIO.cxx#L300
) -> None:
    _check_suffix(path, (".jpg", ".jpeg"))
    writer = sitk.ImageFileWriter()
    writer.SetImageIO(ItkImageIo.JPEG.value)
    writer.SetCompressionLevel(quality)
    writer.SetFileName(str(path))
    writer.Execute(image)


def write_png(
    image: sitk.Image,
    path: TypePath,
) -> None:
    _check_suffix(path, ".png")
    write_image(image, path)


def _check_suffix(path: TypePath, suffixes: str | tuple[str, ...]) -> None:
    if isinstance(suffixes, str):
        suffixes = (suffixes,)
    suffix = Path(path).suffix
    if suffix not in suffixes:
        msg = f'Expected path "{path}" to have a suffix in "{suffixes}"'
        raise ValueError(msg)


def check_quality(value: int) -> int:
    """Check that a value is a valid quality for JPEG compression."""
    min_quality = 0
    max_quality = 100
    if not (min_quality <= value <= max_quality):
        message = f"Quality must be an integer between 0 and 100 but got {value}."
        raise ValueError(message)
    return value
