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
class ExtractOrientations(python.Task["ExtractOrientations.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.extract_orientations import ExtractOrientations

    """

    in_file: File
    indices: list
    in_bvec_file: File

    class Outputs(python.Outputs):
        out_file: File
        out_bvec: list

    @staticmethod
    def function(
        in_file: File, indices: list, in_bvec_file: File
    ) -> tuples[File, list]:
        out_file = attrs.NOTHING
        out_bvec = attrs.NOTHING
        from nipype.utils.filemanip import fname_presuffix

        out_file = fname_presuffix(
            in_file,
            suffix="_subset",
            newpath=os.getcwd(),
        )

        out_file = out_file

        img = nb.load(in_file)
        bzeros = np.squeeze(np.asanyarray(img.dataobj)[..., indices])

        hdr = img.header.copy()
        hdr.set_data_shape(bzeros.shape)
        hdr.set_xyzt_units("mm")
        nb.Nifti1Image(bzeros, img.affine, hdr).to_filename(out_file)

        if in_bvec_file is not attrs.NOTHING:
            bvecs = np.loadtxt(in_bvec_file)[:, indices].T
            out_bvec = [tuple(row) for row in bvecs]

        return out_file, out_bvec
