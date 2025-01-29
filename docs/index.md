# ProceX

Medical image datasets such as X-ray scans are typically stored in DICOM files, using signed 16-bit integers to represent pixel values.
Typically, the DICOM files are converted to a different format, such as PNG or JPEG, and the pixel values are mapped to the full range of the output format and data type.
There is not a standard method to preprocess the DICOM files to a format that can be used to train machine learning models.
This causes variation across methods used in research works in the literature, which are often not described in detail and therefore not reproducible.

ProceX is a Python package that provides a simple interface for converting DICOM files to a format that can be easily read and processed for machine learning tasks, encouraging best practices such as minimizing data loss and file size, and standardization across the field.

ProceX is built on top of [SimpleITK](https://simpleitk.org/) and may be used to enhance image contrast using different methods, resize images with different interpolatin techniques, and convert to a different format, all at once.
Additionally, images can be processed in parallel using multiple CPU cores.

ProceX also includes transforms to normalize pixel values of loaded images stored in 16 bits, so they are ready to be fed into a machine learning model.
