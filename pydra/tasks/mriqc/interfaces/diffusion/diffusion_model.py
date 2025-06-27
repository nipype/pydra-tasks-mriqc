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
class DiffusionModel(python.Task["DiffusionModel.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.diffusion_model import DiffusionModel

    """

    in_file: File
    bvals: list
    bvec_file: File
    brain_mask: File
    decimals: int = 3
    n_shells: int

    class Outputs(python.Outputs):
        out_fa: File
        out_fa_nans: File
        out_fa_degenerate: File
        out_cfa: File
        out_md: File

    @staticmethod
    def function(
        in_file: File,
        bvals: list,
        bvec_file: File,
        brain_mask: File,
        decimals: int,
        n_shells: int,
    ) -> tuples[File, File, File, File, File]:
        out_fa = attrs.NOTHING
        out_fa_nans = attrs.NOTHING
        out_fa_degenerate = attrs.NOTHING
        out_cfa = attrs.NOTHING
        out_md = attrs.NOTHING
        from dipy.core.gradients import gradient_table_from_bvals_bvecs
        from nipype.utils.filemanip import fname_presuffix

        bvals = np.array(bvals)

        gtab = gradient_table_from_bvals_bvecs(
            bvals=bvals,
            bvecs=np.loadtxt(bvec_file).T,
        )

        img = nb.load(in_file)
        data = img.get_fdata(dtype="float32")

        brainmask = np.ones_like(data[..., 0], dtype=bool)

        if brain_mask is not attrs.NOTHING:
            brainmask = (
                np.round(
                    nb.load(brain_mask).get_fdata(),
                    3,
                )
                > 0.5
            )

        if n_shells == 1:
            from dipy.reconst.dti import TensorModel as Model
        else:
            from dipy.reconst.dki import DiffusionKurtosisModel as Model

        fwdtifit = Model(gtab).fit(
            data,
            mask=brainmask,
        )

        fa_data = fwdtifit.fa
        fa_nan_msk = np.isnan(fa_data)
        fa_data[fa_nan_msk] = 0

        fa_data = np.round(fa_data, decimals)
        degenerate_msk = (fa_data < 0) | (fa_data > 1.0)

        fa_data = np.clip(fa_data, 0, 1)

        fa_nii = nb.Nifti1Image(
            fa_data,
            img.affine,
            None,
        )

        fa_nii.header.set_xyzt_units("mm")
        fa_nii.header.set_intent("estimate", name="Fractional Anisotropy (FA)")

        out_fa = fname_presuffix(
            in_file,
            suffix="fa",
            newpath=os.getcwd(),
        )

        fa_nii.to_filename(out_fa)

        fa_nan_nii = nb.Nifti1Image(
            fa_nan_msk.astype(np.uint8),
            img.affine,
            None,
        )

        fa_nan_nii.header.set_xyzt_units("mm")
        fa_nan_nii.header.set_intent("estimate", name="NaNs in the FA map mask")
        fa_nan_nii.header["cal_max"] = 1
        fa_nan_nii.header["cal_min"] = 0

        out_fa_nans = fname_presuffix(
            in_file,
            suffix="desc-fanans_mask",
            newpath=os.getcwd(),
        )
        fa_nan_nii.to_filename(out_fa_nans)

        fa_degenerate_nii = nb.Nifti1Image(
            degenerate_msk.astype(np.uint8),
            img.affine,
            None,
        )

        fa_degenerate_nii.header.set_xyzt_units("mm")
        fa_degenerate_nii.header.set_intent(
            "estimate", name="degenerate vectors in the FA map mask"
        )
        fa_degenerate_nii.header["cal_max"] = 1
        fa_degenerate_nii.header["cal_min"] = 0

        out_fa_degenerate = fname_presuffix(
            in_file,
            suffix="desc-fadegenerate_mask",
            newpath=os.getcwd(),
        )
        fa_degenerate_nii.to_filename(out_fa_degenerate)

        cfa_data = fwdtifit.color_fa
        cfa_nii = nb.Nifti1Image(
            np.clip(cfa_data, a_min=0.0, a_max=1.0),
            img.affine,
            None,
        )

        cfa_nii.header.set_xyzt_units("mm")
        cfa_nii.header.set_intent("estimate", name="Fractional Anisotropy (FA)")
        cfa_nii.header["cal_max"] = 1.0
        cfa_nii.header["cal_min"] = 0.0

        out_cfa = fname_presuffix(
            in_file,
            suffix="cfa",
            newpath=os.getcwd(),
        )
        cfa_nii.to_filename(out_cfa)

        out_md = fname_presuffix(
            in_file,
            suffix="md",
            newpath=os.getcwd(),
        )
        md_data = np.array(fwdtifit.md, dtype="float32")
        md_data[np.isnan(md_data)] = 0
        md_data = np.clip(md_data, 0, 1)
        md_hdr = fa_nii.header.copy()
        md_hdr.set_intent("estimate", name="Mean diffusivity (MD)")
        nb.Nifti1Image(md_data, img.affine, md_hdr).to_filename(out_md)

        return out_fa, out_fa_nans, out_fa_degenerate, out_cfa, out_md
