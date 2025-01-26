"""Low-level image processing operations."""

import numpy as np
import SimpleITK as sitk


def rgb2gray(image: sitk.Image) -> sitk.Image:
    """Convert an RGB image to grayscale.

    If the input image is a single channel image, it is returned as is. If the
    input image is a multi-channel image, the channels are assumed to contain
    the same data, and the first channel is returned as a single channel image.

    Args:
        image: The input image with 3 channels.

    Returns:
        The output image with a single channel.

    Raises:
        ValueError: If the channels contain different data.
    """
    num_channels = image.GetNumberOfComponentsPerPixel()
    if num_channels == 1:
        return image
    channels = [
        sitk.VectorIndexSelectionCast(image, channel) for channel in range(num_channels)
    ]
    # Verify all channels contain the same data
    for channel in channels[1:]:
        if sitk.Hash(channel) != sitk.Hash(channels[0]):
            msg = "RGB images are expected to have identical channels"
            raise ValueError(msg)
    return channels[0]


def squeeze(image: sitk.Image) -> sitk.Image:
    """Remove singleton dimensions from the image.

    Args:
        image: The input image.
    """
    try:
        singleton_dim = image.GetSize().index(1)
    except ValueError:
        return image
    slices = [slice(None)] * image.GetDimension()
    slices[singleton_dim] = 0
    return image[slices]


def enhance_contrast(
    image: sitk.Image,
    *,
    num_bits: int | None = None,
    percentiles: tuple[float, float] = (0, 100),
    values: tuple[float, float] | None = None,
    histeq: bool = False,
) -> sitk.Image:
    """Stretch the intensity range of an image.

    Args:
        image: Input image.
        num_bits: Number of bits used to represent the image intensity.
        percentiles: Lower and upper percentiles to clip the image intensity.
        values: Lower and upper values to clip the image intensity.
        histeq: Whether to perform histogram equalization instead of intensity
            range stretching.

    Returns:
        The output image with the intensity range stretched.

    Raises:
        NotImplementedError: If the pixel type is not supported.
    """
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
    image = sitk.Cast(image, sitk.sitkFloat32)
    if histeq:
        stretched = _histogram_equalization(image, num_bits)
    else:
        if values is not None:
            image = sitk.Clamp(image, image.GetPixelID(), *values)
        if percentiles != (0, 100):
            image = _clip(image, percentiles)
        minimum = 0
        maximum = 2**num_bits - 1
        stretched = sitk.RescaleIntensity(image, minimum, maximum)
    out_dtype = getattr(sitk, f"sitkUInt{num_bits}")
    return sitk.Cast(stretched, out_dtype)


def _clip(image: sitk.Image, percentiles: tuple[float, float]) -> sitk.Image:
    """Clip the intensity range of an image.

    Args:
        image: The input image.
        percentiles: The lower and upper percentiles used to compute the output
            intensity range.

    Returns:
        The output image with the intensity range clipped.
    """
    lower, upper = np.percentile(sitk.GetArrayFromImage(image), percentiles)
    return sitk.Clamp(image, image.GetPixelID(), lower, upper)


def _histogram_equalization(image: sitk.Image, num_bits: int) -> sitk.Image:
    """Perform histogram equalization on the image.

    Args:
        image: The input image.
        num_bits: The number of bits used to represent the image intensity.

    Returns:
        The output image with the intensity range equalized.
    """
    reference_values = np.arange(0, 2**num_bits - 1)
    reference_values_image = reference_values.reshape(1, -1)
    reference = sitk.GetImageFromArray(reference_values_image)
    reference = sitk.Cast(reference, image.GetPixelID())
    return sitk.HistogramMatching(image, reference)


def _get_smoothing_variance(downsampling_factor: float) -> float:
    """Compute the variance for smoothing an image before downsampling.

    The method is presented in
    https://link.springer.com/chapter/10.1007/978-3-319-24571-3_81

    Args:
        downsampling_factor: The factor by which the image is downsampled.

    Returns:
        The variance used for smoothing the image.
    """
    return (downsampling_factor**2 - 1) * (2 * np.sqrt(2 * np.log(2))) ** (-2)


def _smooth(image: sitk.Image, downsampling_factor: float) -> sitk.Image:
    """Smooth the image before downsampling.

    Args:
        image: The input image.
        downsampling_factor: The factor by which the image is downsampled.

    Returns:
        The smoothed image.
    """
    image = sitk.Cast(image, sitk.sitkFloat32)
    variance = _get_smoothing_variance(downsampling_factor)
    num_dims = image.GetDimension()
    variances = [variance] * num_dims
    return sitk.DiscreteGaussian(image, variances, useImageSpacing=False)


def resize(
    image: sitk.Image,
    size: int,
    *,
    interpolator: int = sitk.sitkBSpline,
    smooth: bool = True,
    keep_aspect_ratio: bool = True,
) -> sitk.Image:
    """Resize an image to a specified size.

    Args:
        image: The input image.
        size: The size of the output image.
        interpolator: The interpolation method.
        smooth: Whether to smooth the image before downsampling.
        keep_aspect_ratio: Whether to keep the aspect ratio of the image.

    Returns:
        The output image resized to the specified size.
    """
    if not keep_aspect_ratio:
        msg = "Non-uniform scaling is not supported yet"
        raise NotImplementedError(msg)

    input_size = image.GetSize()
    image_spacing = image.GetSpacing()
    max_dim = max(input_size)
    scale_factor = max_dim / size
    input_dtype = image.GetPixelID()
    input_min, input_max = sitk.MinimumMaximum(image)

    if smooth and scale_factor < 1:
        image = _smooth(image, scale_factor)

    new_size = np.array(input_size) / scale_factor
    new_size = np.round(new_size).astype(int).tolist()
    new_spacing = np.array(image_spacing) * scale_factor

    resized = sitk.Resample(
        image,
        size=new_size,
        interpolator=interpolator,
        outputSpacing=tuple(new_spacing),
        outputOrigin=image.GetOrigin(),
        outputDirection=image.GetDirection(),
    )
    # Clamp the intensity values to the original range as some interpolators
    # may produce out-of-range values
    resized = sitk.Clamp(resized, input_dtype, input_min, input_max)
    return sitk.Cast(resized, input_dtype)
