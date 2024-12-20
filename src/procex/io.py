import enum
from pathlib import Path

import SimpleITK as sitk  # noqa: N813

from .functional import rgb2gray
from .functional import squeeze as squeeze_image
from .typing import TypePath


@enum.unique
class ItkImageIo(str, enum.Enum):
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


def check_uint8(func):
    def wrapper(image, *args, **kwargs):
        if image.GetPixelID() != sitk.sitkUInt8:
            msg = (
                'Expected image to have pixel type "8-bit unsigned integer",'
                f' but got "{image.GetPixelIDTypeAsString()}"'
            )
            raise ValueError(msg)
        return func(image, *args, **kwargs)

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


def _check_suffix(path: TypePath, suffixes: str | tuple[str]) -> None:
    if isinstance(suffixes, str):
        suffixes = (suffixes,)
    suffix = Path(path).suffix
    if suffix not in suffixes:
        msg = f'Expected path "{path}" to have a suffix in "{suffixes}"'
        raise ValueError(msg)
