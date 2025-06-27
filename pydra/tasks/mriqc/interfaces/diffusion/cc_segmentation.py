import attrs
from fileformats.generic import File
import logging
import nibabel as nb
from pydra.tasks.mriqc.nipype_ports.utils.filemanip import fname_presuffix
import numpy as np
import os
from pydra.compose import python
import scipy.ndimage as nd


logger = logging.getLogger(__name__)


@python.define
class CCSegmentation(python.Task["CCSegmentation.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.cc_segmentation import CCSegmentation

    """

    in_fa: File
    in_cfa: File
    min_rgb: tuple = (0.4, 0.008, 0.008)
    max_rgb: tuple = (1.1, 0.25, 0.25)
    wm_threshold: float = 0.35
    clean_mask: bool = False

    class Outputs(python.Outputs):
        out_mask: File
        wm_mask: File
        wm_finalmask: File

    @staticmethod
    def function(
        in_fa: File,
        in_cfa: File,
        min_rgb: tuple,
        max_rgb: tuple,
        wm_threshold: float,
        clean_mask: bool,
    ) -> tuples[File, File, File]:
        out_mask = attrs.NOTHING
        wm_mask = attrs.NOTHING
        wm_finalmask = attrs.NOTHING
        from skimage.measure import label

        out_mask = fname_presuffix(
            in_cfa,
            suffix="ccmask",
            newpath=os.getcwd(),
        )
        wm_mask = fname_presuffix(
            in_cfa,
            suffix="wmmask",
            newpath=os.getcwd(),
        )
        wm_finalmask = fname_presuffix(
            in_cfa,
            suffix="wmfinalmask",
            newpath=os.getcwd(),
        )

        fa_nii = nb.load(in_fa)
        fa_data = np.round(fa_nii.get_fdata(dtype="float32"), 4)
        fa_labels = label((fa_data > wm_threshold).astype(np.uint8))
        wm_mask = fa_labels == np.argmax(np.bincount(fa_labels.flat)[1:]) + 1

        wm_mask_nii = nb.Nifti1Image(
            wm_mask.astype(np.uint8),
            fa_nii.affine,
            None,
        )
        wm_mask_nii.header.set_xyzt_units("mm")
        wm_mask_nii.header.set_intent(
            "estimate", name="white-matter mask (FA thresholded)"
        )
        wm_mask_nii.header["cal_max"] = 1
        wm_mask_nii.header["cal_min"] = 0
        wm_mask_nii.to_filename(wm_mask)

        struct = nd.generate_binary_structure(wm_mask.ndim, wm_mask.ndim - 1)

        wm_mask = nd.grey_closing(
            fa_data,
            structure=struct,
        )
        wm_mask = nd.grey_opening(
            wm_mask,
            structure=struct,
        )

        fa_labels = label((np.round(wm_mask, 4) > wm_threshold).astype(np.uint8))
        wm_mask = fa_labels == np.argmax(np.bincount(fa_labels.flat)[1:]) + 1

        wm_mask_nii = nb.Nifti1Image(
            wm_mask.astype(np.uint8),
            fa_nii.affine,
            wm_mask_nii.header,
        )
        wm_mask_nii.header.set_intent(
            "estimate", name="white-matter mask after binary opening"
        )
        wm_mask_nii.to_filename(wm_finalmask)

        cfa_data = np.round(nb.load(in_cfa).get_fdata(dtype="float32"), 4)
        for i in range(cfa_data.shape[-1]):
            cfa_data[..., i] = nd.grey_closing(
                cfa_data[..., i],
                structure=struct,
            )
            cfa_data[..., i] = nd.grey_opening(
                cfa_data[..., i],
                structure=struct,
            )

        cc_mask = segment_corpus_callosum(
            in_cfa=cfa_data,
            mask=wm_mask,
            min_rgb=min_rgb,
            max_rgb=max_rgb,
            clean_mask=clean_mask,
        )
        cc_mask_nii = nb.Nifti1Image(
            cc_mask.astype(np.uint8),
            fa_nii.affine,
            None,
        )
        cc_mask_nii.header.set_xyzt_units("mm")
        cc_mask_nii.header.set_intent("estimate", name="corpus callosum mask")
        cc_mask_nii.header["cal_max"] = 1
        cc_mask_nii.header["cal_min"] = 0
        cc_mask_nii.to_filename(out_mask)

        return out_mask, wm_mask, wm_finalmask


def segment_corpus_callosum(
    in_cfa: np.ndarray,
    mask: np.ndarray,
    min_rgb: tuple[float, float, float] = (0.6, 0.0, 0.0),
    max_rgb: tuple[float, float, float] = (1.0, 0.1, 0.1),
    clean_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segments the corpus callosum (CC) from a color FA map.

    Parameters
    ----------
    in_cfa : :obj:`~numpy.ndarray`
        The color FA (cFA) map.
    mask : :obj:`~numpy.ndarray` (bool, 3D)
        A white matter mask used to define the initial bounding box.
    min_rgb : :obj:`tuple`, optional
        Minimum RGB values.
    max_rgb : :obj:`tuple`, optional
        Maximum RGB values.
    clean_mask : :obj:`bool`, optional
        Whether the CC mask is finally cleaned-up for spurious off voxels with
        :obj:`dipy.segment.mask.clean_cc_mask`

    Returns
    -------
    cc_mask: :obj:`~numpy.ndarray`
        The final binary mask of the segmented CC.

    Notes
    -----
    This implementation was derived from
    :obj:`dipy.segment.mask.segment_from_cfa`.

    """
    from dipy.segment.mask import bounding_box

    # Prepare a bounding box of the CC
    cc_box = np.zeros_like(mask, dtype=bool)
    mins, maxs = bounding_box(mask)  # mask needs to be volume
    mins = np.array(mins)
    maxs = np.array(maxs)
    diff = (maxs - mins) // 5
    bounds_min = mins + diff
    bounds_max = maxs - diff
    cc_box[
        bounds_min[0] : bounds_max[0],
        bounds_min[1] : bounds_max[1],
        bounds_min[2] : bounds_max[2],
    ] = True
    min_rgb = np.array(min_rgb)
    max_rgb = np.array(max_rgb)
    # Threshold color FA
    cc_mask = np.all(
        (in_cfa >= min_rgb[None, :]) & (in_cfa <= max_rgb[None, :]),
        axis=-1,
    )
    # Apply bounding box and WM mask
    cc_mask *= cc_box & mask
    struct = nd.generate_binary_structure(cc_mask.ndim, cc_mask.ndim - 1)
    # Perform a closing followed by opening operations on the FA.
    cc_mask = nd.binary_closing(
        cc_mask,
        structure=struct,
    )
    cc_mask = nd.binary_opening(
        cc_mask,
        structure=struct,
    )
    if clean_mask:
        from dipy.segment.mask import clean_cc_mask

        cc_mask = clean_cc_mask(cc_mask)
    return cc_mask
