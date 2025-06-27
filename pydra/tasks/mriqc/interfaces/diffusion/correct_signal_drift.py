import attrs
from fileformats.generic import File
import logging
import nibabel as nb
from pydra.tasks.mriqc.nipype_ports.utils.filemanip import fname_presuffix
import numpy as np
import os
from pydra.compose import python


logger = logging.getLogger(__name__)


@python.define
class CorrectSignalDrift(python.Task["CorrectSignalDrift.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.correct_signal_drift import CorrectSignalDrift

    """

    in_file: File
    bias_file: File
    brainmask_file: File
    b0_ixs: list
    bval_file: File
    full_epi: File

    class Outputs(python.Outputs):
        out_file: File
        out_full_file: File
        b0_drift: list
        signal_drift: list

    @staticmethod
    def function(
        in_file: File,
        bias_file: File,
        brainmask_file: File,
        b0_ixs: list,
        bval_file: File,
        full_epi: File,
    ) -> tuples[File, File, list, list]:
        out_file = attrs.NOTHING
        out_full_file = attrs.NOTHING
        b0_drift = attrs.NOTHING
        signal_drift = attrs.NOTHING
        from mriqc import config

        bvals = np.loadtxt(bval_file)
        len_dmri = bvals.size

        img = nb.load(in_file)
        data = img.get_fdata()
        bmask = np.ones_like(data[..., 0], dtype=bool)

        if bias_file is not attrs.NOTHING:
            data *= nb.load(bias_file).get_fdata()[..., np.newaxis]

        if brainmask_file is not attrs.NOTHING:
            bmask = np.round(nb.load(brainmask_file).get_fdata(), 2) > 0.5

        out_file = fname_presuffix(in_file, suffix="_nodrift", newpath=os.getcwd())

        if (b0len := int(data.ndim < 4)) or (b0len := data.shape[3]) < 3:
            logger.warn(
                f"Insufficient number of low-b orientations ({b0len}) "
                "to safely calculate signal drift."
            )

            img.__class__(
                np.round(data.astype("float32"), 4),
                img.affine,
                img.header,
            ).to_filename(out_file)

            if full_epi is not attrs.NOTHING:
                out_full_file = full_epi

            b0_drift = [1.0] * b0len
            signal_drift = [1.0] * len_dmri

        global_signal = np.array(
            [np.median(data[..., n_b0][bmask]) for n_b0 in range(img.shape[-1])]
        ).astype("float32")

        global_signal /= global_signal[0]
        b0_drift = [round(float(gs), 4) for gs in global_signal]

        logger.info(
            f"Correcting drift with {len(global_signal)} b=0 volumes, with "
            "global signal estimated at "
            f'{", ".join([str(v) for v in b0_drift])}.'
        )

        data *= 1.0 / global_signal[np.newaxis, np.newaxis, np.newaxis, :]

        img.__class__(
            data.astype(img.header.get_data_dtype()),
            img.affine,
            img.header,
        ).to_filename(out_file)

        K, A_log = np.polyfit(b0_ixs, np.log(global_signal), 1)

        t_points = np.arange(len_dmri, dtype=int)
        fitted = np.squeeze(_exp_func(t_points, np.exp(A_log), K, 0))
        signal_drift = fitted.astype(float).tolist()

        if full_epi is not attrs.NOTHING:
            out_full_file = fname_presuffix(
                full_epi, suffix="_nodriftfull", newpath=os.getcwd()
            )
            full_img = nb.load(full_epi)
            full_img.__class__(
                full_img.get_fdata() * fitted[np.newaxis, np.newaxis, np.newaxis, :],
                full_img.affine,
                full_img.header,
            ).to_filename(out_full_file)

        return out_file, out_full_file, b0_drift, signal_drift


def _exp_func(t, A, K, C):

    return A * np.exp(K * t) + C
