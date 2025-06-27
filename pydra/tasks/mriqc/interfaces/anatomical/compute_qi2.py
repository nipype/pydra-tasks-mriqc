import attrs
from fileformats.generic import File
import logging
from pydra.tasks.mriqc.qc.anatomical import art_qi2
import nibabel as nb
from pydra.compose import python


logger = logging.getLogger(__name__)


@python.define
class ComputeQI2(python.Task["ComputeQI2.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.anatomical.compute_qi2 import ComputeQI2

    """

    in_file: File
    air_msk: File

    class Outputs(python.Outputs):
        qi2: float
        out_file: File

    @staticmethod
    def function(in_file: File, air_msk: File) -> tuples[float, File]:
        qi2 = attrs.NOTHING
        out_file = attrs.NOTHING
        imdata = nb.load(in_file).get_fdata()
        airdata = nb.load(air_msk).get_fdata()
        qi2, out_file = art_qi2(imdata, airdata)
        qi2 = qi2
        out_file = out_file

        return qi2, out_file
