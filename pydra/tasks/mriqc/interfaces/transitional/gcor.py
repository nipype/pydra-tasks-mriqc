import attrs
from fileformats.generic import File
from fileformats.medimage import Nifti1
import logging
from pydra.compose import shell


logger = logging.getLogger(__name__)


def _list_outputs(inputs=None, stdout=None, stderr=None, output_dir=None):
    inputs = attrs.asdict(inputs)

    return {"out": parsed_inputs["_gcor"]}


def out_callable(output_dir, inputs, stdout, stderr):
    outputs = _list_outputs(
        output_dir=output_dir, inputs=inputs, stdout=stdout, stderr=stderr
    )
    return outputs.get("out")


@shell.define
class GCOR(shell.Task["GCOR.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from fileformats.medimage import Nifti1
    >>> from pydra.tasks.mriqc.interfaces.transitional.gcor import GCOR

    >>> task = GCOR()
    >>> task.inputs.in_file = Nifti1.mock("func.nii")
    >>> task.inputs.mask = File.mock()
    >>> task.inputs.nfirst = 4
    >>> task.cmdline
    '@compute_gcor -nfirst 4 -input func.nii'


    """

    executable = "@compute_gcor"
    in_file: Nifti1 = shell.arg(
        help="input dataset to compute the GCOR over",
        argstr="-input {in_file}",
        position=-1,
    )
    mask: File = shell.arg(
        help="mask dataset, for restricting the computation", argstr="-mask {mask}"
    )
    nfirst: int = shell.arg(
        help="specify number of initial TRs to ignore", argstr="-nfirst {nfirst}"
    )
    no_demean: bool = shell.arg(
        help="do not (need to) demean as first step", argstr="-no_demean"
    )

    class Outputs(shell.Outputs):
        out: float | None = shell.out(
            help="global correlation value", callable=out_callable
        )
