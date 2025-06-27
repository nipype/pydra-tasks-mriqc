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
class SplitShells(python.Task["SplitShells.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.split_shells import SplitShells

    """

    in_file: File
    bvals: list

    class Outputs(python.Outputs):
        out_file: List

    @staticmethod
    def function(in_file: File, bvals: list) -> List:
        out_file = attrs.NOTHING
        from nipype.utils.filemanip import fname_presuffix

        bval_list = np.rint(bvals).astype(int)
        bvals = np.unique(bval_list)
        img = nb.load(in_file)
        data = np.asanyarray(img.dataobj)

        out_file = []

        for bval in bvals:
            fname = fname_presuffix(
                in_file, suffix=f"_b{bval:05d}", newpath=os.getcwd()
            )
            out_file.append(fname)

            img.__class__(
                data[..., np.argwhere(bval_list == bval)],
                img.affine,
                img.header,
            ).to_filename(fname)

        return out_file
