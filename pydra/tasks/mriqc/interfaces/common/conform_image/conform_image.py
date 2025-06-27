import attrs
from fileformats.generic import File
import logging
import nibabel as nib
import numpy as np
from os import path as op
from pydra.compose import python


logger = logging.getLogger(__name__)


@python.define
class ConformImage(python.Task["ConformImage.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.common.conform_image.conform_image import ConformImage

    """

    in_file: File
    check_ras: bool = True
    check_dtype: bool = True

    class Outputs(python.Outputs):
        out_file: File

    @staticmethod
    def function(in_file: File, check_ras: bool, check_dtype: bool) -> File:
        out_file = attrs.NOTHING
        """
        Execute this interface with the provided runtime.
    
        TODO: Is the *runtime* argument required? It doesn't seem to be used
              anywhere.
    
        Parameters
        ----------
        runtime : Any
            Execution runtime ?
    
        Returns
        -------
        Any
            Execution runtime ?
        """

        nii = nib.squeeze_image(nib.load(in_file))

        if check_ras:
            nii = nib.as_closest_canonical(nii)

        if check_dtype:
            nii = _check_dtype(nii, in_file=in_file)

        out_file, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            out_file, ext2 = op.splitext(out_file)
            ext = ext2 + ext
        out_file_name = OUT_FILE.format(prefix=out_file, ext=ext)
        out_file = op.abspath(out_file_name)
        nii.to_filename(out_file)

        return out_file


def _check_dtype(nii: nib.Nifti1Image, in_file=None) -> nib.Nifti1Image:
    """
    Checks the NIfTI header datatype and converts the data to the matching
    numpy dtype.

    Parameters
    ----------
    nii : nib.Nifti1Image
        Input image

    Returns
    -------
    nib.Nifti1Image
        Converted input image
    """
    header = nii.header.copy()
    datatype = int(header["datatype"])
    _warn_suspicious_dtype(datatype, in_file=in_file)
    try:
        dtype = NUMPY_DTYPE[datatype]
    except KeyError:
        return nii
    else:
        header.set_data_dtype(dtype)
        converted = np.asanyarray(nii.dataobj, dtype=dtype)
        return nib.Nifti1Image(converted, nii.affine, header)


def _warn_suspicious_dtype(dtype: int, in_file=None) -> None:
    """
    Warns about binary type *nii* images.

    Parameters
    ----------
    dtype : int
        NIfTI header datatype
    """
    if dtype == 1:
        dtype_message = "Input image {in_file} has a suspicious data type: '{dtype}'".format(
            in_file=in_file, dtype=dtype
        )
        logger.warning(dtype_message)


NUMPY_DTYPE = {
    1: np.uint8,
    2: np.uint8,
    4: np.uint16,
    8: np.uint32,
    64: np.float32,
    256: np.uint8,
    1024: np.uint32,
    1280: np.uint32,
    1536: np.float32,
}

OUT_FILE = "{prefix}_conformed{ext}"
