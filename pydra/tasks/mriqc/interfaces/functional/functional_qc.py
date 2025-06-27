import attrs
from fileformats.generic import File
import logging
from pydra.tasks.mriqc.qc.anatomical import efc, fber, snr, summary_stats
from pydra.tasks.mriqc.qc.functional import gsr
from pydra.tasks.mriqc.utils.misc import _flatten_dict
import nibabel as nb
import numpy as np
from pydra.compose import python
import typing as ty


logger = logging.getLogger(__name__)


@python.define
class FunctionalQC(python.Task["FunctionalQC.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.functional.functional_qc import FunctionalQC

    """

    in_epi: File
    in_hmc: File
    in_tsnr: File
    in_mask: File
    direction: ty.Any = all
    in_fd: File
    fd_thres: float = 0.2
    in_dvars: File
    in_fwhm: list

    class Outputs(python.Outputs):
        fber: float
        efc: float
        snr: float
        gsr: dict
        tsnr: float
        dvars: dict
        fd: dict
        fwhm: dict
        size: dict
        spacing: dict
        summary: dict
        out_qc: dict

    @staticmethod
    def function(
        in_epi: File,
        in_hmc: File,
        in_tsnr: File,
        in_mask: File,
        direction: ty.Any,
        in_fd: File,
        fd_thres: float,
        in_dvars: File,
        in_fwhm: list,
    ) -> tuples[
        float, float, float, dict, float, dict, dict, dict, dict, dict, dict, dict
    ]:
        fber = attrs.NOTHING
        efc = attrs.NOTHING
        snr = attrs.NOTHING
        gsr = attrs.NOTHING
        tsnr = attrs.NOTHING
        dvars = attrs.NOTHING
        fd = attrs.NOTHING
        fwhm = attrs.NOTHING
        size = attrs.NOTHING
        spacing = attrs.NOTHING
        summary = attrs.NOTHING
        out_qc = attrs.NOTHING
        self_dict = {}

        epinii = nb.load(in_epi)
        epidata = np.nan_to_num(np.float32(epinii.dataobj))
        epidata[epidata < 0] = 0

        hmcnii = nb.load(in_hmc)
        hmcdata = np.nan_to_num(np.float32(hmcnii.dataobj))
        hmcdata[hmcdata < 0] = 0

        msknii = nb.load(in_mask)
        mskdata = np.asanyarray(msknii.dataobj) > 0
        if np.sum(mskdata) < 100:
            raise RuntimeError(
                "Detected less than 100 voxels belonging to the brain mask. "
                "MRIQC failed to process this dataset."
            )

        rois = {"fg": mskdata.astype(np.uint8), "bg": (~mskdata).astype(np.uint8)}
        stats = summary_stats(epidata, rois)
        summary = stats

        snr = snr(stats["fg"]["median"], stats["fg"]["stdv"], stats["fg"]["n"])

        fber = fber(epidata, mskdata.astype(np.uint8))

        efc = efc(epidata)

        gsr = {}
        if direction == "all":
            epidir = ["x", "y"]
        else:
            epidir = [direction]

        for axis in epidir:
            gsr[axis] = gsr(epidata, mskdata.astype(np.uint8), direction=axis)

        dvars_avg = np.loadtxt(in_dvars, skiprows=1, usecols=list(range(3))).mean(
            axis=0
        )
        dvars_col = ["std", "nstd", "vstd"]
        dvars = {key: float(val) for key, val in zip(dvars_col, dvars_avg)}

        tsnr_data = nb.load(in_tsnr).get_fdata()
        tsnr = float(np.median(tsnr_data[mskdata]))

        fd_data = np.loadtxt(in_fd, skiprows=1)
        num_fd = (fd_data > fd_thres).sum()
        fd = {
            "mean": float(fd_data.mean()),
            "num": int(num_fd),
            "perc": float(num_fd * 100 / (len(fd_data) + 1)),
        }

        fwhm = np.array(in_fwhm[:3]) / np.array(hmcnii.header.get_zooms()[:3])
        fwhm = {
            "x": float(fwhm[0]),
            "y": float(fwhm[1]),
            "z": float(fwhm[2]),
            "avg": float(np.average(fwhm)),
        }

        size = {
            "x": int(hmcdata.shape[0]),
            "y": int(hmcdata.shape[1]),
            "z": int(hmcdata.shape[2]),
        }
        spacing = {
            i: float(v) for i, v in zip(["x", "y", "z"], hmcnii.header.get_zooms()[:3])
        }

        try:
            size["t"] = int(hmcdata.shape[3])
        except IndexError:
            pass

        try:
            spacing["tr"] = float(hmcnii.header.get_zooms()[3])
        except IndexError:
            pass

        out_qc = _flatten_dict(self_dict["_results"])

        return (
            fber,
            efc,
            snr,
            gsr,
            tsnr,
            dvars,
            fd,
            fwhm,
            size,
            spacing,
            summary,
            out_qc,
        )
