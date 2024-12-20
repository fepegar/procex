import numpy as np
import SimpleITK as sitk  # noqa: N813


def rgb2gray(image: sitk.Image) -> sitk.Image:
    num_channels = image.GetNumberOfComponentsPerPixel()
    if num_channels == 1:
        return image
    channels = [
        sitk.VectorIndexSelectionCast(image, channel)
        for channel in range(num_channels)
    ]
    # TODO(fernando): verify that all channels contain the same information?
    return channels[0]


def squeeze(image: sitk.Image) -> sitk.Image:
    # TODO(fernando): what if image has multiple singleton dimensions?
    try:
        singleton_dim = image.GetSize().index(1)
    except ValueError:
        return image
    slices = [slice(None)] * image.GetDimension()
    slices[singleton_dim] = 0
    return image[slices]


# TODO(fernando): add percentiles
def stretch(
    image: sitk.Image,
    num_bits: int | None = None,
    percentiles: tuple[float, float] = (0, 100),
    *,
    histeq: bool = False,
) -> sitk.Image:
    if num_bits is None:
        match image.GetPixelIDValue():
            case sitk.sitkUInt8:
                num_bits = 8
            case sitk.sitkUInt16:
                num_bits = 16
            case _:
                pixel_type_string = image.GetPixelIDTypeAsString()
                msg = f'Unsupported pixel type "{pixel_type_string}"'
                raise NotImplementedError(msg)
    minimum = 0
    maximum = 2 ** num_bits - 1
    if histeq:
        stretched = _histogram_equalization(image, num_bits)
    else:
        if percentiles != (0, 100):
            image = clip(image, percentiles)
        stretched = sitk.RescaleIntensity(image, minimum, maximum)
    out_dtype = getattr(sitk, f"sitkUInt{num_bits}")
    return sitk.Cast(stretched, out_dtype)


def clip(image: sitk.Image, percentiles: tuple[float, float]) -> sitk.Image:
    lower, upper = np.percentile(sitk.GetArrayFromImage(image), percentiles)
    return sitk.Clamp(image, image.GetPixelID(), lower, upper)


def _histogram_equalization(image: sitk.Image, num_bits: int) -> sitk.Image:
    reference_values = np.arange(0, 2**num_bits-1)
    reference_values_image = reference_values.reshape(1, -1)
    reference = sitk.GetImageFromArray(reference_values_image)
    reference = sitk.Cast(reference, image.GetPixelID())
    return sitk.HistogramMatching(image, reference)


def _get_smoothing_variance(downsampling_factor: float) -> float:
    # https://link.springer.com/chapter/10.1007/978-3-319-24571-3_81
    return (downsampling_factor ** 2 - 1) * (2 * np.sqrt(2 * np.log(2))) ** (-2)


def _smooth(image: sitk.Image, downsampling_factor: float) -> sitk.Image:
    image = sitk.Cast(image, sitk.sitkFloat32)
    variance = _get_smoothing_variance(downsampling_factor)
    return sitk.DiscreteGaussian(image, variance, useImageSpacing=False)


def resize(
    image: sitk.Image,
    size: int,
    interpolator: int = sitk.sitkBSpline,
    *,
    smooth: bool = True,
    keep_aspect_ratio: bool = True,
) -> sitk.Image:
    if not keep_aspect_ratio:
        msg = "Non-uniform scaling is not yet supported"
        raise NotImplementedError(msg)

    input_size = image.GetSize()
    image_spacing = image.GetSpacing()
    max_dim = max(input_size)
    scale_factor = max_dim / size

    if smooth:
        image = _smooth (image, scale_factor)

    new_size = np.array(input_size) / scale_factor
    new_size = np.round(new_size).astype(int).tolist()
    new_spacing = np.array(image_spacing) * scale_factor
    interpolator: int = sitk.sitkBSpline

    return sitk.Resample(
        image,
        size=new_size,
        interpolator=interpolator,
        outputSpacing=tuple(new_spacing),
        outputOrigin=image.GetOrigin(),
        outputDirection=image.GetDirection(),
    )
