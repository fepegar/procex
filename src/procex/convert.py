import numpy.typing as npt
import SimpleITK as sitk  # noqa: N813
from PIL import Image

from .functional import stretch as stretch_image


def to_pil(image: sitk.Image, *, stretch: bool = False, **kwargs) -> Image.Image:
    if stretch:
        image = stretch_image(image, **kwargs)
    array = sitk.GetArrayViewFromImage(image)
    return Image.fromarray(array)


def to_numpy(image: sitk.Image) -> npt.NDArray:
    return sitk.GetArrayViewFromImage(image)
