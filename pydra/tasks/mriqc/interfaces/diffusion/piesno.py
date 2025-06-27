import attrs
from fileformats.generic import File
import logging
import nibabel as nb
from pydra.tasks.mriqc.nipype_ports.utils.filemanip import fname_presuffix
import numpy as np
import os
from pydra.compose import python


logger = logging.getLogger(__name__)


@python.define
class PIESNO(python.Task["PIESNO.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.piesno import PIESNO

    """

    in_file: File
    n_channels: int = 4

    class Outputs(python.Outputs):
        sigma: float
        out_mask: File

    @staticmethod
    def function(in_file: File, n_channels: int) -> tuples[float, File]:
        sigma = attrs.NOTHING
        out_mask = attrs.NOTHING
        out_mask = fname_presuffix(
            in_file,
            suffix="piesno",
            newpath=os.getcwd(),
        )

        in_nii = nb.load(in_file)
        data = np.round(in_nii.get_fdata(), 4).astype("float32")

        sigma, maskdata = noise_piesno(data)

        header = in_nii.header.copy()
        header.set_data_dtype(np.uint8)
        header.set_xyzt_units("mm")
        header.set_intent("estimate", name="PIESNO noise voxels mask")
        header["cal_max"] = 1
        header["cal_min"] = 0

        nb.Nifti1Image(
            maskdata.astype(np.uint8),
            in_nii.affine,
            header,
        ).to_filename(out_mask)

        sigma = round(float(np.median(sigma)), 5)

        return sigma, out_mask


def noise_piesno(data: np.ndarray, n_channels: int = 4) -> (np.ndarray, np.ndarray):
    """
    Estimates noise in raw diffusion MRI (dMRI) data using the PIESNO algorithm.

    This function implements the PIESNO (Probabilistic Identification and Estimation
    of Noise) algorithm [Koay2009]_ to estimate the standard deviation (sigma) of the
    noise in each voxel of a 4D dMRI data array. The PIESNO algorithm assumes Rician
    distributed signal and exploits the statistical properties of the noise to
    separate it from the underlying signal.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The 4D raw dMRI data array.
    n_channels : :obj:`int`, optional (default=4)
        The number of diffusion-encoding channels in the data. This value is used
        internally by the PIESNO algorithm.

    Returns
    -------
    sigma : :obj:`~numpy.ndarray`
        The estimated noise standard deviation for each voxel in the data array.
    mask : :obj:`~numpy.ndarray`
        A brain mask estimated by PIESNO. This mask identifies voxels containing
        mostly noise and can be used for further processing.

    """
    from dipy.denoise.noise_estimate import piesno

    sigma, mask = piesno(data, N=n_channels, return_mask=True)
    return sigma, mask
