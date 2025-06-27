import attrs
from fileformats.generic import File
import logging
from pydra.tasks.mriqc.nipype_ports.utils.misc import normalize_mc_params
import numpy as np
import os.path as op
from pathlib import Path
from pydra.compose import python
from pydra.tasks.mriqc.nipype_ports.utils.misc import normalize_mc_params
import typing as ty


logger = logging.getLogger(__name__)


@python.define
class FramewiseDisplacement(python.Task["FramewiseDisplacement.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.nipype_ports.algorithms.confounds.framewise_displacement import FramewiseDisplacement

    """

    in_file: File
    parameter_source: ty.Any
    radius: float = 50
    out_file: Path = fd_power_2012.txt
    out_figure: Path = fd_power_2012.pdf
    series_tr: float
    save_plot: bool = False
    normalize: bool = False
    figdpi: int = 100
    figsize: ty.Any = (11.7, 2.3)

    class Outputs(python.Outputs):
        out_file: File
        out_figure: File
        fd_average: float

    @staticmethod
    def function(
        in_file: File,
        parameter_source: ty.Any,
        radius: float,
        out_file: Path,
        out_figure: Path,
        series_tr: float,
        save_plot: bool,
        normalize: bool,
        figdpi: int,
        figsize: ty.Any,
    ) -> tuples[File, File, float]:
        out_file = attrs.NOTHING
        out_figure = attrs.NOTHING
        fd_average = attrs.NOTHING
        self_dict = {}
        mpars = np.loadtxt(in_file)  # mpars is N_t x 6
        mpars = np.apply_along_axis(
            func1d=normalize_mc_params,
            axis=1,
            arr=mpars,
            source=parameter_source,
        )
        diff = mpars[:-1, :6] - mpars[1:, :6]
        diff[:, 3:6] *= radius
        fd_res = np.abs(diff).sum(axis=1)

        self_dict["_results"] = {
            "out_file": op.abspath(out_file),
            "fd_average": float(fd_res.mean()),
        }
        np.savetxt(out_file, fd_res, header="FramewiseDisplacement", comments="")

        if save_plot:
            tr = None
            if series_tr is not attrs.NOTHING:
                tr = series_tr

            if normalize and tr is None:
                IFLOGGER.warning("FD plot cannot be normalized if TR is not set")

            out_figure = op.abspath(out_figure)
            fig = plot_confound(
                fd_res,
                figsize,
                "FD",
                units="mm",
                series_tr=tr,
                normalize=normalize,
            )
            fig.savefig(
                out_figure,
                dpi=float(figdpi),
                format=out_figure[-3:],
                bbox_inches="tight",
            )
            fig.clf()

        return out_file, out_figure, fd_average


IFLOGGER = logging.getLogger("nipype.interface")
