import attrs
from fileformats.generic import File
import logging
import nibabel as nb
import numpy as np
import os
from pathlib import Path
from pydra.compose import python
import scipy.ndimage as nd


logger = logging.getLogger(__name__)


@python.define
class ArtifactMask(python.Task["ArtifactMask.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.anatomical.artifact_mask import ArtifactMask

    """

    in_file: File
    head_mask: File
    glabella_xyz: tuple = (0.0, 90.0, -14.0)
    inion_xyz: tuple = (0.0, -120.0, -14.0)
    ind2std_xfm: File
    zscore: float = 10.0

    class Outputs(python.Outputs):
        out_hat_msk: File
        out_art_msk: File
        out_air_msk: File

    @staticmethod
    def function(
        in_file: File,
        head_mask: File,
        glabella_xyz: tuple,
        inion_xyz: tuple,
        ind2std_xfm: File,
        zscore: float,
    ) -> tuples[File, File, File]:
        out_hat_msk = attrs.NOTHING
        out_art_msk = attrs.NOTHING
        out_air_msk = attrs.NOTHING
        from nibabel.affines import apply_affine
        from nitransforms.linear import Affine

        in_file = Path(in_file)
        imnii = nb.as_closest_canonical(nb.load(in_file))
        imdata = np.nan_to_num(imnii.get_fdata().astype(np.float32))

        xfm = Affine.from_filename(ind2std_xfm, fmt="itk")

        ras2ijk = np.linalg.inv(imnii.affine)
        glabella_ijk, inion_ijk = apply_affine(
            ras2ijk, xfm.map([glabella_xyz, inion_xyz])
        )

        hmdata = np.bool_(nb.load(head_mask).dataobj)

        dist = nd.morphology.distance_transform_edt(~hmdata)

        hmdata[:, :, : int(inion_ijk[2])] = 1
        hmdata[:, (hmdata.shape[1] // 2) :, : int(glabella_ijk[2])] = 1

        dist[~hmdata] = 0
        dist /= dist.max()

        qi1_img = artifact_mask(imdata, (~hmdata), dist, zscore=zscore)

        fname = in_file.relative_to(in_file.parent).stem
        ext = "".join(in_file.suffixes)

        outdir = Path(os.getcwd()).absolute()
        out_hat_msk = str(outdir / f"{fname}_hat{ext}")
        out_art_msk = str(outdir / f"{fname}_art{ext}")
        out_air_msk = str(outdir / f"{fname}_air{ext}")

        hdr = imnii.header.copy()
        hdr.set_data_dtype(np.uint8)
        imnii.__class__(qi1_img.astype(np.uint8), imnii.affine, hdr).to_filename(
            out_art_msk
        )

        airdata = (~hmdata).astype(np.uint8)
        imnii.__class__(airdata, imnii.affine, hdr).to_filename(out_hat_msk)

        airdata[qi1_img > 0] = 0
        imnii.__class__(airdata.astype(np.uint8), imnii.affine, hdr).to_filename(
            out_air_msk
        )

        return out_hat_msk, out_art_msk, out_air_msk


def artifact_mask(imdata, airdata, distance, zscore=10.0):
    """Compute a mask of artifacts found in the air region."""
    from statsmodels.robust.scale import mad

    qi1_msk = np.zeros(imdata.shape, dtype=bool)
    bg_data = imdata[airdata]
    if (bg_data > 0).sum() < 10:
        return qi1_msk
    # Standardize the distribution of the background
    bg_spread = mad(bg_data[bg_data > 0])
    bg_data[bg_data > 0] = bg_data[bg_data > 0] / bg_spread
    # Apply this threshold to the background voxels to identify voxels
    # contributing artifacts.
    qi1_msk[airdata] = bg_data > zscore
    qi1_msk[distance < 0.10] = False
    # Create a structural element to be used in an opening operation.
    struct = nd.generate_binary_structure(3, 1)
    qi1_msk = nd.binary_opening(qi1_msk, struct).astype(np.uint8)
    return qi1_msk
