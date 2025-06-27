import attrs
from fileformats.generic import Directory, File
import json
import logging
import numpy as np
from pathlib import Path
from pydra.compose import python, shell, workflow
from .functional_qc import FunctionalQC
from .gather_timeseries import GatherTimeseries
from .select_echo import SelectEcho
from .spikes import Spikes
from pydra.utils.typing import MultiInputObj
import typing as ty
import yaml


logger = logging.getLogger(__name__)


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


def _get_echotime(inlist):

    if isinstance(inlist, list):
        retval = [_get_echotime(el) for el in inlist]
        return retval[0] if len(retval) == 1 else retval
    echo_time = inlist.get("EchoTime", None) if inlist else None
    if echo_time:
        return float(echo_time)


def _robust_zscore(data):

    return (data - np.atleast_2d(np.median(data, axis=1)).T) / np.atleast_2d(
        data.std(axis=1)
    ).T


def find_peaks(data):

    t_z = [data[:, :, i, :].mean(axis=0).mean(axis=0) for i in range(data.shape[2])]
    return t_z


def find_spikes(data, spike_thresh):

    data -= np.median(np.median(np.median(data, axis=0), axis=0), axis=0)
    slice_mean = np.median(np.median(data, axis=0), axis=0)
    t_z = _robust_zscore(slice_mean)
    spikes = np.abs(t_z) > spike_thresh
    spike_inds = np.transpose(spikes.nonzero())
    # mask out the spikes and recompute z-scores using variance uncontaminated with spikes.
    # This will catch smaller spikes that may have been swamped by big
    # ones.
    data.mask[:, :, spike_inds[:, 0], spike_inds[:, 1]] = True
    slice_mean2 = np.median(np.median(data, axis=0), axis=0)
    t_z = _robust_zscore(slice_mean2)
    spikes = np.logical_or(spikes, np.abs(t_z) > spike_thresh)
    spike_inds = [tuple(i) for i in np.transpose(spikes.nonzero())]
    return spike_inds, t_z


def select_echo(
    in_files: str | list[str],
    te_echos: list[float | type(attrs.NOTHING) | None] | None = None,
    te_reference: float = 0.030,
) -> str:
    """
    Select the echo file with the closest echo time to the reference echo time.

    Used to grab the echo file when processing multi-echo data through workflows
    that only accept a single file.

    Parameters
    ----------
    in_files : :obj:`str` or :obj:`list`
        A single filename or a list of filenames.
    te_echos : :obj:`list` of :obj:`float`
        List of echo times corresponding to each file.
        If not a number (typically, a :obj:`~nipype.interfaces.base.type(attrs.NOTHING)`),
        the function selects the second echo.
    te_reference : float, optional
        Reference echo time used to find the closest echo time.

    Returns
    -------
    str
        The selected echo file.

    Examples
    --------
    >>> select_echo("single-echo.nii.gz")
    ('single-echo.nii.gz', -1)

    >>> select_echo(["single-echo.nii.gz"])
    ('single-echo.nii.gz', -1)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ... )
    ('echo2.nii.gz', 1)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ...     te_echos=[12.5, 28.5, 34.2, 45.0, 56.1, 68.4],
    ...     te_reference=33.1,
    ... )
    ('echo3.nii.gz', 2)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ...     te_echos=[12.5, 28.5, 34.2, 45.0, 56.1],
    ...     te_reference=33.1,
    ... )
    ('echo2.nii.gz', 1)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ...     te_echos=[12.5, 28.5, 34.2, 45.0, 56.1, None],
    ...     te_reference=33.1,
    ... )
    ('echo2.nii.gz', 1)

    """
    if not isinstance(in_files, (list, tuple)):
        return in_files, -1
    if len(in_files) == 1:
        return in_files[0], -1
    import numpy as np

    n_echos = len(in_files)
    if te_echos is not None and len(te_echos) == n_echos:
        try:
            index = np.argmin(np.abs(np.array(te_echos) - te_reference))
            return in_files[index], index
        except TypeError:
            pass
    return in_files[1], 1
