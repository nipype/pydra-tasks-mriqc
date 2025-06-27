import attrs
from fileformats.generic import File
import logging
import nibabel as nb
from pydra.tasks.mriqc.nipype_ports.utils.filemanip import fname_presuffix
import numpy as np
from pydra.compose import python
import scipy.ndimage as nd


logger = logging.getLogger(__name__)


@python.define
class Harmonize(python.Task["Harmonize.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.anatomical.harmonize import Harmonize

    """

    in_file: File
    wm_mask: File
    brain_mask: File
    erodemsk: bool = True
    thresh: float = 0.9
    min_size: int = 30

    class Outputs(python.Outputs):
        out_file: File

    @staticmethod
    def function(
        in_file: File,
        wm_mask: File,
        brain_mask: File,
        erodemsk: bool,
        thresh: float,
        min_size: int,
    ) -> File:
        out_file = attrs.NOTHING
        in_file = nb.load(in_file)
        data = in_file.get_fdata()

        wm_mask = nb.load(wm_mask).get_fdata()
        wm_mask[wm_mask < thresh] = 0
        wm_mask[wm_mask > 0] = 1
        wm_mask = wm_mask.astype(bool)
        wm_mask_size = wm_mask.sum()

        if wm_mask_size < min_size:
            brain_mask = nb.load(brain_mask).get_fdata() > 0.5
            wm_mask = brain_mask.copy()
            wm_mask[data < np.percentile(data[brain_mask], 75)] = False
            wm_mask[data > np.percentile(data[brain_mask], 95)] = False
        elif erodemsk:

            struct = nd.generate_binary_structure(3, 2)

            wm_mask = nd.binary_erosion(
                wm_mask.astype(np.uint8), structure=struct
            ).astype(bool)

        data *= 1000.0 / np.median(data[wm_mask])

        out_file = fname_presuffix(in_file, suffix="_harmonized", newpath=".")
        in_file.__class__(data, in_file.affine, in_file.header).to_filename(out_file)

        out_file = out_file

        return out_file
