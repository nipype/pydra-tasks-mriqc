import attrs
from fileformats.generic import File
import logging
import nibabel as nb
import numpy as np
import os.path as op
from pathlib import Path
from pydra.compose import python
import typing as ty


logger = logging.getLogger(__name__)


@python.define
class TSNR(python.Task["TSNR.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.nipype_ports.algorithms.confounds.tsnr import TSNR

    """

    in_file: List
    regress_poly: ty.Any
    tsnr_file: Path = tsnr.nii.gz
    mean_file: Path = mean.nii.gz
    stddev_file: Path = stdev.nii.gz
    detrended_file: Path = detrend.nii.gz

    class Outputs(python.Outputs):
        tsnr_file: File
        mean_file: File
        stddev_file: File
        detrended_file: File

    @staticmethod
    def function(
        in_file: List,
        regress_poly: ty.Any,
        tsnr_file: Path,
        mean_file: Path,
        stddev_file: Path,
        detrended_file: Path,
    ) -> tuples[File, File, File, File]:
        tsnr_file = attrs.NOTHING
        mean_file = attrs.NOTHING
        stddev_file = attrs.NOTHING
        detrended_file = attrs.NOTHING
        img = nb.load(in_file[0])
        header = img.header.copy()
        vollist = [nb.load(filename) for filename in in_file]
        data = np.concatenate(
            [
                vol.get_fdata(dtype=np.float32).reshape(vol.shape[:3] + (-1,))
                for vol in vollist
            ],
            axis=3,
        )
        data = np.nan_to_num(data)

        if data.dtype.kind == "i":
            header.set_data_dtype(np.float32)
            data = data.astype(np.float32)

        if regress_poly is not attrs.NOTHING:
            data = regress_poly(regress_poly, data, remove_mean=False)[0]
            img = nb.Nifti1Image(data, img.affine, header)
            nb.save(img, op.abspath(detrended_file))

        meanimg = np.mean(data, axis=3)
        stddevimg = np.std(data, axis=3)
        tsnr = np.zeros_like(meanimg)
        stddevimg_nonzero = stddevimg > 1.0e-3
        tsnr[stddevimg_nonzero] = (
            meanimg[stddevimg_nonzero] / stddevimg[stddevimg_nonzero]
        )
        img = nb.Nifti1Image(tsnr, img.affine, header)
        nb.save(img, op.abspath(tsnr_file))
        img = nb.Nifti1Image(meanimg, img.affine, header)
        nb.save(img, op.abspath(mean_file))
        img = nb.Nifti1Image(stddevimg, img.affine, header)
        nb.save(img, op.abspath(stddev_file))
        self_dict = {}
        outputs = {}
        for k in ["tsnr_file", "mean_file", "stddev_file"]:
            outputs[k] = op.abspath(getattr(self_dict["inputs"], k))

        if regress_poly is not attrs.NOTHING:
            detrended_file = op.abspath(detrended_file)

        return tsnr_file, mean_file, stddev_file, detrended_file


IFLOGGER = logging.getLogger("nipype.interface")
