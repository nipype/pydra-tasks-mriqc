import attrs
from fileformats.generic import File
import logging
from pydra.tasks.mriqc.nipype_ports.utils.misc import normalize_mc_params
import numpy as np
from os import path as op
import os
import pandas as pd
from pydra.compose import python
import typing as ty


logger = logging.getLogger(__name__)


@python.define
class GatherTimeseries(python.Task["GatherTimeseries.Outputs"]):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.functional.gather_timeseries import GatherTimeseries

    """

    dvars: File
    fd: File
    mpars: File
    mpars_source: ty.Any
    outliers: File
    quality: File

    class Outputs(python.Outputs):
        timeseries_file: File
        timeseries_metadata: dict

    @staticmethod
    def function(
        dvars: File,
        fd: File,
        mpars: File,
        mpars_source: ty.Any,
        outliers: File,
        quality: File,
    ) -> tuples[File, dict]:
        timeseries_file = attrs.NOTHING
        timeseries_metadata = attrs.NOTHING

        mpars = np.apply_along_axis(
            func1d=normalize_mc_params,
            axis=1,
            arr=np.loadtxt(mpars),  # mpars is N_t x 6
            source=mpars_source,
        )
        timeseries = pd.DataFrame(
            mpars, columns=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
        )

        dvars = pd.read_csv(
            dvars,
            sep=r"\s+",
            skiprows=1,  # column names have spaces
            header=None,
            names=["dvars_std", "dvars_nstd", "dvars_vstd"],
        )
        dvars.index = pd.RangeIndex(1, timeseries.index.max() + 1)

        fd = pd.read_csv(fd, sep=r"\s+", header=0, names=["framewise_displacement"])
        fd.index = pd.RangeIndex(1, timeseries.index.max() + 1)

        aqi = pd.read_csv(quality, sep=r"\s+", header=None, names=["aqi"])

        aor = pd.read_csv(outliers, sep=r"\s+", header=None, names=["aor"])

        timeseries = pd.concat((timeseries, dvars, fd, aqi, aor), axis=1)

        timeseries_file = op.join(os.getcwd(), "timeseries.tsv")

        timeseries.to_csv(timeseries_file, sep="\t", index=False, na_rep="n/a")

        timeseries_file = timeseries_file
        timeseries_metadata = _build_timeseries_metadata()

        return timeseries_file, timeseries_metadata


def _build_timeseries_metadata():

    return {
        "trans_x": {
            "LongName": "Translation Along X Axis",
            "Description": "Estimated Motion Parameter",
            "Units": "mm",
        },
        "trans_y": {
            "LongName": "Translation Along Y Axis",
            "Description": "Estimated Motion Parameter",
            "Units": "mm",
        },
        "trans_z": {
            "LongName": "Translation Along Z Axis",
            "Description": "Estimated Motion Parameter",
            "Units": "mm",
        },
        "rot_x": {
            "LongName": "Rotation Around X Axis",
            "Description": "Estimated Motion Parameter",
            "Units": "rad",
        },
        "rot_y": {
            "LongName": "Rotation Around X Axis",
            "Description": "Estimated Motion Parameter",
            "Units": "rad",
        },
        "rot_z": {
            "LongName": "Rotation Around X Axis",
            "Description": "Estimated Motion Parameter",
            "Units": "rad",
        },
        "dvars_std": {
            "LongName": "Derivative of RMS Variance over Voxels, Standardized",
            "Description": (
                "Indexes the rate of change of BOLD signal across"
                "the entire brain at each frame of data, normalized with the"
                "standard deviation of the temporal difference time series"
            ),
        },
        "dvars_nstd": {
            "LongName": ("Derivative of RMS Variance over Voxels, Non-Standardized"),
            "Description": (
                "Indexes the rate of change of BOLD signal across"
                "the entire brain at each frame of data, not normalized."
            ),
        },
        "dvars_vstd": {
            "LongName": "Derivative of RMS Variance over Voxels, Standardized",
            "Description": (
                "Indexes the rate of change of BOLD signal across"
                "the entire brain at each frame of data, normalized across"
                "time by that voxel standard deviation across time,"
                "before computing the RMS of the temporal difference"
            ),
        },
        "framewise_displacement": {
            "LongName": "Framewise Displacement",
            "Description": (
                "A quantification of the estimated bulk-head"
                "motion calculated using formula proposed by Power (2012)"
            ),
            "Units": "mm",
        },
        "aqi": {
            "LongName": "AFNI's Quality Index",
            "Description": "Mean quality index as computed by AFNI's 3dTqual",
        },
        "aor": {
            "LongName": "AFNI's Fraction of Outliers per Volume",
            "Description": (
                "Mean fraction of outliers per fMRI volume as given by AFNI's 3dToutcount"
            ),
        },
    }
