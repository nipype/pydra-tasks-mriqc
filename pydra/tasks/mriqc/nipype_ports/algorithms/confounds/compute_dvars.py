import attrs
from fileformats.generic import File
import logging
import numpy as np
import os.path as op
from pydra.compose import python
import typing as ty


logger = logging.getLogger(__name__)


@python.define
class ComputeDVARS(python.Task["ComputeDVARS.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.nipype_ports.algorithms.confounds.compute_dvars import ComputeDVARS

    """

    in_file: File
    in_mask: File
    remove_zerovariance: bool = True
    variance_tol: float = 1e-07
    save_std: bool = True
    save_nstd: bool = False
    save_vxstd: bool = False
    save_all: bool = False
    series_tr: float
    save_plot: bool = False
    figdpi: int = 100
    figsize: ty.Any = (11.7, 2.3)
    figformat: ty.Any = png
    intensity_normalization: float = 1000.0

    class Outputs(python.Outputs):
        out_std: File
        out_nstd: File
        out_vxstd: File
        out_all: File
        avg_std: float
        avg_nstd: float
        avg_vxstd: float
        fig_std: File
        fig_nstd: File
        fig_vxstd: File

    @staticmethod
    def function(
        in_file: File,
        in_mask: File,
        remove_zerovariance: bool,
        variance_tol: float,
        save_std: bool,
        save_nstd: bool,
        save_vxstd: bool,
        save_all: bool,
        series_tr: float,
        save_plot: bool,
        figdpi: int,
        figsize: ty.Any,
        figformat: ty.Any,
        intensity_normalization: float,
    ) -> tuples[File, File, File, File, float, float, float, File, File, File]:
        out_std = attrs.NOTHING
        out_nstd = attrs.NOTHING
        out_vxstd = attrs.NOTHING
        out_all = attrs.NOTHING
        avg_std = attrs.NOTHING
        avg_nstd = attrs.NOTHING
        avg_vxstd = attrs.NOTHING
        fig_std = attrs.NOTHING
        fig_nstd = attrs.NOTHING
        fig_vxstd = attrs.NOTHING
        self_dict = {}
        self_dict["_results"] = {}

        dvars = compute_dvars(
            in_file,
            in_mask,
            remove_zerovariance=remove_zerovariance,
            variance_tol=variance_tol,
            intensity_normalization=intensity_normalization,
        )

        (
            avg_std,
            avg_nstd,
            avg_vxstd,
        ) = np.mean(
            dvars, axis=1
        ).astype(float)

        tr = None
        if series_tr is not attrs.NOTHING:
            tr = series_tr

        if save_std:
            out_file = _gen_fname("dvars_std", ext="tsv", in_file=in_file)
            np.savetxt(out_file, dvars[0], fmt="%0.6f")
            out_std = out_file

            if save_plot:
                fig_std = _gen_fname("dvars_std", ext=figformat, in_file=in_file)
                fig = plot_confound(
                    dvars[0], figsize, "Standardized DVARS", series_tr=tr
                )
                fig.savefig(
                    fig_std,
                    dpi=float(figdpi),
                    format=figformat,
                    bbox_inches="tight",
                )
                fig.clf()

        if save_nstd:
            out_file = _gen_fname("dvars_nstd", ext="tsv", in_file=in_file)
            np.savetxt(out_file, dvars[1], fmt="%0.6f")
            out_nstd = out_file

            if save_plot:
                fig_nstd = _gen_fname("dvars_nstd", ext=figformat, in_file=in_file)
                fig = plot_confound(dvars[1], figsize, "DVARS", series_tr=tr)
                fig.savefig(
                    fig_nstd,
                    dpi=float(figdpi),
                    format=figformat,
                    bbox_inches="tight",
                )
                fig.clf()

        if save_vxstd:
            out_file = _gen_fname("dvars_vxstd", ext="tsv", in_file=in_file)
            np.savetxt(out_file, dvars[2], fmt="%0.6f")
            out_vxstd = out_file

            if save_plot:
                fig_vxstd = _gen_fname("dvars_vxstd", ext=figformat, in_file=in_file)
                fig = plot_confound(
                    dvars[2], figsize, "Voxelwise std DVARS", series_tr=tr
                )
                fig.savefig(
                    fig_vxstd,
                    dpi=float(figdpi),
                    format=figformat,
                    bbox_inches="tight",
                )
                fig.clf()

        if save_all:
            out_file = _gen_fname("dvars", ext="tsv", in_file=in_file)
            np.savetxt(
                out_file,
                np.vstack(dvars).T,
                fmt="%0.8f",
                delimiter="\t",
                header="std DVARS\tnon-std DVARS\tvx-wise std DVARS",
                comments="",
            )
            out_all = out_file

        return (
            out_std,
            out_nstd,
            out_vxstd,
            out_all,
            avg_std,
            avg_nstd,
            avg_vxstd,
            fig_std,
            fig_nstd,
            fig_vxstd,
        )


def _gen_fname(suffix, ext=None, in_file=None):
    fname, in_ext = op.splitext(op.basename(in_file))

    if in_ext == ".gz":
        fname, in_ext2 = op.splitext(fname)
        in_ext = in_ext2 + in_ext

    if ext is None:
        ext = in_ext

    if ext.startswith("."):
        ext = ext[1:]

    return op.abspath(f"{fname}_{suffix}.{ext}")


IFLOGGER = logging.getLogger("nipype.interface")
