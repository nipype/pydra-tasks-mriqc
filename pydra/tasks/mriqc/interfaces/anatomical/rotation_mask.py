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
class RotationMask(python.Task["RotationMask.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.anatomical.rotation_mask import RotationMask

    """

    in_file: File

    class Outputs(python.Outputs):
        out_file: File

    @staticmethod
    def function(in_file: File) -> File:
        out_file = attrs.NOTHING
        in_file = nb.load(in_file)
        data = in_file.get_fdata()
        mask = data <= 0

        mask = np.pad(mask, pad_width=(1,), mode="constant", constant_values=1)

        struct = nd.generate_binary_structure(3, 2)
        mask = nd.binary_opening(mask, structure=struct).astype(np.uint8)

        label_im, nb_labels = nd.label(mask)
        if nb_labels > 2:
            sizes = nd.sum(mask, label_im, list(range(nb_labels + 1)))
            ordered = sorted(zip(sizes, list(range(nb_labels + 1))), reverse=True)
            for _, label in ordered[2:]:
                mask[label_im == label] = 0

        mask = mask[1:-1, 1:-1, 1:-1]

        if mask.sum() < 500:
            mask = np.zeros_like(mask, dtype=np.uint8)

        out_img = in_file.__class__(mask, in_file.affine, in_file.header)
        out_img.header.set_data_dtype(np.uint8)

        out_file = fname_presuffix(in_file, suffix="_rotmask", newpath=".")
        out_img.to_filename(out_file)
        out_file = out_file

        return out_file
