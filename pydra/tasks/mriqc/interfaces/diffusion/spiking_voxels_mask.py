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
class SpikingVoxelsMask(python.Task["SpikingVoxelsMask.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.spiking_voxels_mask import SpikingVoxelsMask

    """

    in_file: File
    brain_mask: File
    z_threshold: float = 3.0
    b_masks: list

    class Outputs(python.Outputs):
        out_mask: File

    @staticmethod
    def function(
        in_file: File, brain_mask: File, z_threshold: float, b_masks: list
    ) -> File:
        out_mask = attrs.NOTHING
        out_mask = fname_presuffix(
            in_file,
            suffix="spikesmask",
            newpath=os.getcwd(),
        )

        in_nii = nb.load(in_file)
        data = np.round(in_nii.get_fdata(), 4).astype("float32")

        bmask_nii = nb.load(brain_mask)
        brainmask = np.round(bmask_nii.get_fdata(), 2).astype("float32")

        spikes_mask = get_spike_mask(
            data,
            shell_masks=b_masks,
            brainmask=brainmask,
            z_threshold=z_threshold,
        )

        header = bmask_nii.header.copy()
        header.set_data_dtype(np.uint8)
        header.set_xyzt_units("mm")
        header.set_intent("estimate", name="spiking voxels mask")
        header["cal_max"] = 1
        header["cal_min"] = 0

        spikes_mask_nii = nb.Nifti1Image(
            spikes_mask.astype(np.uint8),
            bmask_nii.affine,
            header,
        )
        spikes_mask_nii.to_filename(out_mask)

        return out_mask


def get_spike_mask(
    data: np.ndarray, shell_masks: list, brainmask: np.ndarray, z_threshold: float = 3.0
) -> np.ndarray:
    """
    Creates a binary mask classifying voxels in the data array as spike or non-spike.

    This function identifies voxels with signal intensities exceeding a threshold based
    on standard deviations above the mean. The threshold can be applied globally to
    the entire data array, or it can be calculated for groups of voxels defined by
    the ``grouping_vals`` parameter.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The data array to be thresholded.
    z_threshold : :obj:`float`, optional (default=3.0)
        The number of standard deviations to use above the mean as the threshold
        multiplier.
    brainmask : :obj:`~numpy.ndarray`
        The brain mask.
    shell_masks : :obj:`list`
        A list of :obj:`~numpy.ndarray` objects

    Returns:
    -------
    spike_mask : :obj:`~numpy.ndarray`
        A binary mask where ``True`` values indicate voxels classified as spikes and
        ``False`` values indicate non-spikes. The mask has the same shape as the input
        data array.

    """
    spike_mask = np.zeros_like(data, dtype=bool)
    brainmask = brainmask >= 0.5
    for b_mask in shell_masks:
        shelldata = data[..., b_mask]
        a_thres = z_threshold * shelldata[brainmask].std() + shelldata[brainmask].mean()
        spike_mask[..., b_mask] = shelldata > a_thres
    return spike_mask
