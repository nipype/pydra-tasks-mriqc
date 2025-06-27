import attrs
from fileformats.generic import File
import logging
import nibabel as nb
from pydra.compose import python


logger = logging.getLogger(__name__)


@python.define
class NonSteadyStateDetector(python.Task["NonSteadyStateDetector.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.nipype_ports.algorithms.confounds.non_steady_state_detector import NonSteadyStateDetector

    """

    in_file: File

    class Outputs(python.Outputs):
        n_volumes_to_discard: int

    @staticmethod
    def function(in_file: File) -> int:
        n_volumes_to_discard = attrs.NOTHING
        self_dict = {}
        in_nii = nb.load(in_file)
        global_signal = (
            in_nii.dataobj[:, :, :, :50].mean(axis=0).mean(axis=0).mean(axis=0)
        )

        self_dict["_results"] = {"n_volumes_to_discard": is_outlier(global_signal)}

        return n_volumes_to_discard
