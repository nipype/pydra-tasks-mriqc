import attrs
from fileformats.generic import File
import logging
from pydra.tasks.mriqc.qc.anatomical import (
    art_qi1,
    cjv,
    cnr,
    efc,
    fber,
    rpve,
    snr,
    snr_dietrich,
    summary_stats,
    volume_fraction,
    wm2max,
)
from pydra.tasks.mriqc.utils.misc import _flatten_dict
import nibabel as nb
import numpy as np
from pydra.compose import python


logger = logging.getLogger(__name__)


@python.define
class StructuralQC(python.Task["StructuralQC.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.anatomical.structural_qc import StructuralQC

    """

    in_file: File
    in_noinu: File
    in_segm: File
    in_bias: File
    head_msk: File
    air_msk: File
    rot_msk: File
    artifact_msk: File
    in_pvms: List
    in_tpms: List
    mni_tpms: List
    in_fwhm: list
    human: bool = True

    class Outputs(python.Outputs):
        summary: dict
        icvs: dict
        rpve: dict
        size: dict
        spacing: dict
        fwhm: dict
        inu: dict
        snr: dict
        snrd: dict
        cnr: float
        fber: float
        efc: float
        qi_1: float
        wm2max: float
        cjv: float
        out_qc: dict
        out_noisefit: File
        tpm_overlap: dict

    @staticmethod
    def function(
        in_file: File,
        in_noinu: File,
        in_segm: File,
        in_bias: File,
        head_msk: File,
        air_msk: File,
        rot_msk: File,
        artifact_msk: File,
        in_pvms: List,
        in_tpms: List,
        mni_tpms: List,
        in_fwhm: list,
        human: bool,
    ) -> tuples[
        dict,
        dict,
        dict,
        dict,
        dict,
        dict,
        dict,
        dict,
        dict,
        float,
        float,
        float,
        float,
        float,
        float,
        dict,
        File,
        dict,
    ]:
        summary = attrs.NOTHING
        icvs = attrs.NOTHING
        rpve = attrs.NOTHING
        size = attrs.NOTHING
        spacing = attrs.NOTHING
        fwhm = attrs.NOTHING
        inu = attrs.NOTHING
        snr = attrs.NOTHING
        snrd = attrs.NOTHING
        cnr = attrs.NOTHING
        fber = attrs.NOTHING
        efc = attrs.NOTHING
        qi_1 = attrs.NOTHING
        wm2max = attrs.NOTHING
        cjv = attrs.NOTHING
        out_qc = attrs.NOTHING
        out_noisefit = attrs.NOTHING
        tpm_overlap = attrs.NOTHING
        self_dict = {}
        imnii = nb.load(in_noinu)

        inudata = np.nan_to_num(imnii.get_fdata())
        inudata[inudata < 0] = 0

        if np.all(inudata < 1e-5):
            raise RuntimeError(
                "Input inhomogeneity-corrected data seem empty. "
                "MRIQC failed to process this dataset."
            )

        segnii = nb.load(in_segm)
        segdata = np.asanyarray(segnii.dataobj).astype(np.uint8)

        if np.sum(segdata > 0) < 1e3:
            raise RuntimeError(
                "Input segmentation data is likely corrupt. MRIQC failed to process this dataset."
            )

        airdata = np.asanyarray(nb.load(air_msk).dataobj).astype(np.uint8)
        artdata = np.asanyarray(nb.load(artifact_msk).dataobj).astype(np.uint8)

        headdata = np.asanyarray(nb.load(head_msk).dataobj).astype(np.uint8)
        if np.sum(headdata > 0) < 100:
            raise RuntimeError(
                "Detected less than 100 voxels belonging to the head mask. "
                "MRIQC failed to process this dataset."
            )

        rotdata = np.asanyarray(nb.load(rot_msk).dataobj).astype(np.uint8)

        pvms = {
            label: nb.load(fname).get_fdata()
            for label, fname in zip(("csf", "gm", "wm"), in_pvms)
        }
        pvmdata = list(pvms.values())

        pvms["bg"] = airdata

        stats = summary_stats(inudata, pvms)
        summary = stats

        snrvals = []
        snr = {}
        for tlabel in ("csf", "wm", "gm"):
            snrvals.append(
                snr(
                    stats[tlabel]["median"],
                    stats[tlabel]["stdv"],
                    stats[tlabel]["n"],
                )
            )
            snr[tlabel] = snrvals[-1]
        snr["total"] = float(np.mean(snrvals))

        snrvals = []
        snrd = {
            tlabel: snr_dietrich(
                stats[tlabel]["median"],
                mad_air=stats["bg"]["mad"],
                sigma_air=stats["bg"]["stdv"],
            )
            for tlabel in ["csf", "wm", "gm"]
        }
        snrd["total"] = float(np.mean([val for _, val in list(snrd.items())]))

        cnr = cnr(
            stats["wm"]["median"],
            stats["gm"]["median"],
            stats["bg"]["stdv"],
            stats["wm"]["stdv"],
            stats["gm"]["stdv"],
        )

        fber = fber(inudata, headdata, rotdata)

        efc = efc(inudata, rotdata)

        wm2max = wm2max(inudata, stats["wm"]["median"])

        qi_1 = art_qi1(airdata, artdata)

        cjv = cjv(
            stats["wm"]["median"],
            stats["gm"]["median"],
            stats["wm"]["mad"],
            stats["gm"]["mad"],
        )

        fwhm = np.array(in_fwhm[:3]) / np.array(imnii.header.get_zooms()[:3])
        fwhm = {
            "x": float(fwhm[0]),
            "y": float(fwhm[1]),
            "z": float(fwhm[2]),
            "avg": float(np.average(fwhm)),
        }

        icvs = volume_fraction(pvmdata)

        rpve = rpve(pvmdata, segdata)

        size = {
            "x": int(inudata.shape[0]),
            "y": int(inudata.shape[1]),
            "z": int(inudata.shape[2]),
        }
        spacing = {
            i: float(v) for i, v in zip(["x", "y", "z"], imnii.header.get_zooms()[:3])
        }

        try:
            size["t"] = int(inudata.shape[3])
        except IndexError:
            pass

        try:
            spacing["tr"] = float(imnii.header.get_zooms()[3])
        except IndexError:
            pass

        bias = nb.load(in_bias).get_fdata()[segdata > 0]
        inu = {
            "range": float(
                np.abs(np.percentile(bias, 95.0) - np.percentile(bias, 5.0))
            ),
            "med": float(np.median(bias)),
        }  # pylint: disable=E1101

        mni_tpms = [nb.load(tpm).get_fdata() for tpm in mni_tpms]
        in_tpms = [nb.load(tpm).get_fdata() for tpm in in_pvms]
        overlap = fuzzy_jaccard(in_tpms, mni_tpms)
        tpm_overlap = {
            "csf": overlap[0],
            "gm": overlap[1],
            "wm": overlap[2],
        }

        out_qc = _flatten_dict(self_dict["_results"])

        return (
            summary,
            icvs,
            rpve,
            size,
            spacing,
            fwhm,
            inu,
            snr,
            snrd,
            cnr,
            fber,
            efc,
            qi_1,
            wm2max,
            cjv,
            out_qc,
            # out_noisefit,
            tpm_overlap,
        )


def fuzzy_jaccard(in_tpms, in_mni_tpms):

    overlaps = []
    for tpm, mni_tpm in zip(in_tpms, in_mni_tpms):
        tpm = tpm.reshape(-1)
        mni_tpm = mni_tpm.reshape(-1)
        num = np.min([tpm, mni_tpm], axis=0).sum()
        den = np.max([tpm, mni_tpm], axis=0).sum()
        overlaps.append(float(num / den))
    return overlaps
