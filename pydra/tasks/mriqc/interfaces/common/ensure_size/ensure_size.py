import attrs
from fileformats.generic import File
import logging
import nibabel as nib
from pydra.tasks.ants.auto import ApplyTransforms
from pydra.tasks.niworkflows.data import Loader
import numpy as np
from os import path as op
from pydra.compose import python


logger = logging.getLogger(__name__)


@python.define
class EnsureSize(python.Task["EnsureSize.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.common.ensure_size.ensure_size import EnsureSize

    """

    in_file: File
    in_mask: File
    pixel_size: float = 2.0

    class Outputs(python.Outputs):
        out_file: File
        out_mask: File

    @staticmethod
    def function(in_file: File, in_mask: File, pixel_size: float) -> tuples[File, File]:
        out_file = attrs.NOTHING
        out_mask = attrs.NOTHING
        nii = nib.load(in_file)
        size_ok = _check_size(nii, pixel_size=pixel_size)
        if size_ok:
            out_file = in_file
            if in_mask is not attrs.NOTHING:
                out_mask = in_mask
        else:

            aff_base = nii.header.get_base_affine()
            aff_base_inv = np.linalg.inv(aff_base)

            center_idx = (np.array(nii.shape[:3]) - 1) * 0.5
            center_mm = aff_base.dot(center_idx.tolist() + [1])

            min_mm = aff_base.dot([-0.5, -0.5, -0.5, 1])
            max_mm = aff_base.dot((np.array(nii.shape[:3]) - 0.5).tolist() + [1])
            extent_mm = np.abs(max_mm - min_mm)[:3]

            new_size = np.array(extent_mm / pixel_size, dtype=int)

            new_base = aff_base[:3, :3] * np.abs(aff_base_inv[:3, :3]) * pixel_size

            new_center_idx = (new_size - 1) * 0.5
            new_affine_base = np.eye(4)
            new_affine_base[:3, :3] = new_base
            new_affine_base[:3, 3] = center_mm[:3] - new_base.dot(new_center_idx)

            rotation = nii.affine.dot(aff_base_inv)
            new_affine = rotation.dot(new_affine_base)

            hdr = nii.header.copy()
            hdr.set_data_shape(new_size)
            nib.Nifti1Image(
                np.zeros(new_size, dtype=nii.get_data_dtype()), new_affine, hdr
            ).to_filename(REF_FILE_NAME)

            out_prefix, ext = op.splitext(op.basename(in_file))
            if ext == ".gz":
                out_prefix, ext2 = op.splitext(out_prefix)
                ext = ext2 + ext

            out_file_name = OUT_FILE_NAME.format(prefix=out_prefix, ext=ext)
            out_file = op.abspath(out_file_name)

            ApplyTransforms(
                dimension=3,
                input_image=in_file,
                reference_image=REF_FILE_NAME,
                interpolation="LanczosWindowedSinc",
                transforms=[str(load_data("data/itk_identity.tfm").absolute())],
                output_image=out_file,
            ).run()

            out_file = out_file

            if in_mask is not attrs.NOTHING:
                hdr = nii.header.copy()
                hdr.set_data_shape(new_size)
                hdr.set_data_dtype(np.uint8)
                nib.Nifti1Image(
                    np.zeros(new_size, dtype=np.uint8), new_affine, hdr
                ).to_filename(REF_MASK_NAME)

                out_mask_name = OUT_MASK_NAME.format(prefix=out_prefix, ext=ext)
                out_mask = op.abspath(out_mask_name)
                ApplyTransforms(
                    dimension=3,
                    input_image=in_mask,
                    reference_image=REF_MASK_NAME,
                    interpolation="NearestNeighbor",
                    transforms=[str(load_data("data/itk_identity.tfm").absolute())],
                    output_image=out_mask,
                ).run()

                out_mask = out_mask

        return out_file, out_mask


def _check_size(nii: nib.Nifti1Image, pixel_size=None) -> bool:
    zooms = nii.header.get_zooms()
    size_diff = np.array(zooms[:3]) - (pixel_size - 0.1)
    if np.all(size_diff >= -1e-3):
        logger.info('Voxel size is large enough.')
        return True
    else:
        small_voxel_message = 'One or more voxel dimensions (%f, %f, %f) are smaller than the requested voxel size (%f) - diff=(%f, %f, %f)'.format(
            *zooms[:3], pixel_size, *size_diff
        )
        logger.info(small_voxel_message)
        return False


OUT_FILE_NAME = "{prefix}_resampled{ext}"

OUT_MASK_NAME = "{prefix}_resmask{ext}"

REF_FILE_NAME = "resample_ref.nii.gz"

REF_MASK_NAME = "mask_ref.nii.gz"

load_data = Loader("pydra.tasks.mriqc")
