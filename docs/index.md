# ProceX

Medical image datasets such as X-ray scans are typically stored in DICOM files, using signed 16-bit integers to represent pixel values.
There is not an obvious or standard method to convert these pixel values to a format that can be used by machine learning models.
Typically, the DICOM files are converted to a different format, such as PNG or JPEG, and the pixel values are normalized to the range of the output format.

ProceX is a Python package that provides a simple interface for converting DICOM files to a format that can be easily read and processed for machine learning tasks, encouraging best practices such as minimizing data loss and file size.

ProceX is built on top of [SimpleITK](https://simpleitk.org/) and may be used to enhance image contrast using different methods, resize images with different interpolatin techniques, and convert to a different format, all at once.
Additionally, images can be processed in parallel using multiple CPU cores.
