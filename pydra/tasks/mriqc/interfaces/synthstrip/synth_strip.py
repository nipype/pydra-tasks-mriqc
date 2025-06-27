from fileformats.generic import File
import logging
from pathlib import Path
from pydra.compose import shell


logger = logging.getLogger(__name__)


@shell.define
class SynthStrip(shell.Task["SynthStrip.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.synthstrip.synth_strip import SynthStrip

    """

    executable = "synthstrip"
    in_file: File = shell.arg(
        help="Input image to be brain extracted",
        argstr="-i {in_file}",
        copy_mode="File.CopyMode.copy",
    )
    use_gpu: bool = shell.arg(help="Use GPU", argstr="-g", default=False)
    model: File = shell.arg(
        help="file containing model's weights",
        argstr="--model {model}",
        default="/Applications/freesurfer/7.4.1/models/synthstrip.1.pt",
    )
    border_mm: int = shell.arg(
        help="Mask border threshold in mm", argstr="-b {border_mm}", default=1
    )
    num_threads: int = shell.arg(help="Number of threads", argstr="-n {num_threads}")

    class Outputs(shell.Outputs):
        out_mask: Path = shell.outarg(
            help="store brainmask to file",
            argstr="-m {out_mask}",
            path_template="{in_file}_desc-brain_mask.nii.gz",
        )
