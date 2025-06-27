import attrs
from fileformats.generic import File
import logging
import nibabel as nb
from pydra.tasks.mriqc.nipype_ports.utils.filemanip import fname_presuffix
import numpy as np
import os
from pydra.compose import python
import typing as ty


logger = logging.getLogger(__name__)


@python.define
class WeightedStat(python.Task["WeightedStat.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.weighted_stat import WeightedStat

    """

    in_file: File
    in_weights: list
    stat: ty.Any = mean

    class Outputs(python.Outputs):
        out_file: File

    @staticmethod
    def function(in_file: File, in_weights: list, stat: ty.Any) -> File:
        out_file = attrs.NOTHING
        img = nb.load(in_file)
        weights = [float(w) for w in in_weights]
        data = np.asanyarray(img.dataobj)
        statmap = np.average(data, weights=weights, axis=-1)

        out_file = fname_presuffix(in_file, suffix=f"_{stat}", newpath=os.getcwd())

        if stat == "std":
            statmap = np.sqrt(
                np.average(
                    (data - statmap[..., np.newaxis]) ** 2, weights=weights, axis=-1
                )
            )

        hdr = img.header.copy()
        img.__class__(
            statmap.astype(hdr.get_data_dtype()),
            img.affine,
            hdr,
        ).to_filename(out_file)

        return out_file
