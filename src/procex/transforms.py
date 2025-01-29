"""Transforms to preprocess images for training and inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from PIL import Image


class ToTensor:
    """Convert an 8- or 16-bit Pillow image to a tensor and normalize to [0, 1]."""
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Convert, cast and normalize the image.

        Args:
            image: The input image.

        Returns:
            The image as a tensor with shape (C, H, W) and dtype torch.float32.

        Raises:
            ImportError: If torch or torchvision are not installed.
            ValueError: If the image dtype is not np.uint8 or np.uint16.
        """
        try:
            import torch
            from torchvision.transforms import _functional_pil as F_pil
        except ImportError as e:
            message = (
                "ToTensorEightOrSixteenBits requires extra packages to be installed."
                " Install with `pip install procex[torch]`."
            )
            raise ImportError(message) from e

        array = np.array(image)
        match array.dtype:
            case np.uint8:
                num_bits = 8
            case np.uint16:
                num_bits = 16
            case _:
                message = f"Unsupported dtype: {array.dtype}"
                raise ValueError(message)
        array = array.astype(np.float32) / 2 ** num_bits - 1
        # The following lines were adapted from
        # https://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html#to_tensor
        tensor = torch.from_numpy(array)
        num_channels = F_pil.get_image_num_channels(image)
        tensor = tensor.view(image.size[1], image.size[0], num_channels)
        return tensor.permute((2, 0, 1)).contiguous()
