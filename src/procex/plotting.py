import matplotlib.pyplot as plt
import SimpleITK as sitk  # noqa: N813


def plot_image(image: sitk.Image, ax: plt.Axes = None) -> None:
    if ax is None:
        _, ax = plt.subplots()
    array = sitk.GetArrayViewFromImage(image)
    ax.imshow(array, cmap="gray")


def plot_histogram(
    image: sitk.Image,
    ax: plt.Axes = None,
    *,
    log: bool = True,
) -> None:
    if ax is None:
        _, ax = plt.subplots()
    array = sitk.GetArrayViewFromImage(image)
    ax.hist(array.ravel(), bins=256)
    if log:
        ax.set_yscale("log")
