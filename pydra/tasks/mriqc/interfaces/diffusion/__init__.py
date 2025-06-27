import attrs
from fileformats.generic import Directory, File
import json
import logging
import numpy as np
from pathlib import Path
from pydra.compose import python, shell, workflow
from .cc_segmentation import CCSegmentation
from .correct_signal_drift import CorrectSignalDrift
from .diffusion_model import DiffusionModel
from .diffusion_qc import DiffusionQC
from .extract_orientations import ExtractOrientations
from .filter_shells import FilterShells
from .number_of_shells import NumberOfShells
from .piesno import PIESNO
from .read_dwi_metadata import ReadDWIMetadata
from .rotate_vectors import RotateVectors
from .spiking_voxels_mask import SpikingVoxelsMask
from .split_shells import SplitShells
from .weighted_stat import WeightedStat
from pydra.utils.typing import MultiInputObj
import scipy.ndimage as nd
import typing as ty
import yaml


logger = logging.getLogger(__name__)


def _exp_func(t, A, K, C):

    return A * np.exp(K * t) + C


def _rms(estimator, X):
    """
    Callable to pass to GridSearchCV that will calculate a distance score.

    To consider: using `MDL
    <https://erikerlandson.github.io/blog/2016/08/03/x-medoids-using-minimum-description-length-to-identify-the-k-in-k-medoids/>`__

    """
    if len(np.unique(estimator.cluster_centers_)) < estimator.n_clusters:
        return -np.inf
    # Calculate distance from assigned shell centroid
    distance = X - estimator.cluster_centers_[estimator.predict(X)]
    # Make negative so CV optimizes minimizes the error
    return -np.sqrt(distance**2).sum()


def get_spike_mask(
    data: np.ndarray, shell_masks: list, brainmask: np.ndarray, z_threshold: float = 3.0
) -> np.ndarray:
    """
    Creates a binary mask classifying voxels in the data array as spike or non-spike.

    This function identifies voxels with signal intensities exceeding a threshold based
    on standard deviations above the mean. The threshold can be applied globally to
    the entire data array, or it can be calculated for groups of voxels defined by
    the ``grouping_vals`` parameter.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The data array to be thresholded.
    z_threshold : :obj:`float`, optional (default=3.0)
        The number of standard deviations to use above the mean as the threshold
        multiplier.
    brainmask : :obj:`~numpy.ndarray`
        The brain mask.
    shell_masks : :obj:`list`
        A list of :obj:`~numpy.ndarray` objects

    Returns:
    -------
    spike_mask : :obj:`~numpy.ndarray`
        A binary mask where ``True`` values indicate voxels classified as spikes and
        ``False`` values indicate non-spikes. The mask has the same shape as the input
        data array.

    """
    spike_mask = np.zeros_like(data, dtype=bool)
    brainmask = brainmask >= 0.5
    for b_mask in shell_masks:
        shelldata = data[..., b_mask]
        a_thres = z_threshold * shelldata[brainmask].std() + shelldata[brainmask].mean()
        spike_mask[..., b_mask] = shelldata > a_thres
    return spike_mask


def noise_piesno(data: np.ndarray, n_channels: int = 4) -> (np.ndarray, np.ndarray):
    """
    Estimates noise in raw diffusion MRI (dMRI) data using the PIESNO algorithm.

    This function implements the PIESNO (Probabilistic Identification and Estimation
    of Noise) algorithm [Koay2009]_ to estimate the standard deviation (sigma) of the
    noise in each voxel of a 4D dMRI data array. The PIESNO algorithm assumes Rician
    distributed signal and exploits the statistical properties of the noise to
    separate it from the underlying signal.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The 4D raw dMRI data array.
    n_channels : :obj:`int`, optional (default=4)
        The number of diffusion-encoding channels in the data. This value is used
        internally by the PIESNO algorithm.

    Returns
    -------
    sigma : :obj:`~numpy.ndarray`
        The estimated noise standard deviation for each voxel in the data array.
    mask : :obj:`~numpy.ndarray`
        A brain mask estimated by PIESNO. This mask identifies voxels containing
        mostly noise and can be used for further processing.

    """
    from dipy.denoise.noise_estimate import piesno

    sigma, mask = piesno(data, N=n_channels, return_mask=True)
    return sigma, mask


def segment_corpus_callosum(
    in_cfa: np.ndarray,
    mask: np.ndarray,
    min_rgb: tuple[float, float, float] = (0.6, 0.0, 0.0),
    max_rgb: tuple[float, float, float] = (1.0, 0.1, 0.1),
    clean_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segments the corpus callosum (CC) from a color FA map.

    Parameters
    ----------
    in_cfa : :obj:`~numpy.ndarray`
        The color FA (cFA) map.
    mask : :obj:`~numpy.ndarray` (bool, 3D)
        A white matter mask used to define the initial bounding box.
    min_rgb : :obj:`tuple`, optional
        Minimum RGB values.
    max_rgb : :obj:`tuple`, optional
        Maximum RGB values.
    clean_mask : :obj:`bool`, optional
        Whether the CC mask is finally cleaned-up for spurious off voxels with
        :obj:`dipy.segment.mask.clean_cc_mask`

    Returns
    -------
    cc_mask: :obj:`~numpy.ndarray`
        The final binary mask of the segmented CC.

    Notes
    -----
    This implementation was derived from
    :obj:`dipy.segment.mask.segment_from_cfa`.

    """
    from dipy.segment.mask import bounding_box

    # Prepare a bounding box of the CC
    cc_box = np.zeros_like(mask, dtype=bool)
    mins, maxs = bounding_box(mask)  # mask needs to be volume
    mins = np.array(mins)
    maxs = np.array(maxs)
    diff = (maxs - mins) // 5
    bounds_min = mins + diff
    bounds_max = maxs - diff
    cc_box[
        bounds_min[0] : bounds_max[0],
        bounds_min[1] : bounds_max[1],
        bounds_min[2] : bounds_max[2],
    ] = True
    min_rgb = np.array(min_rgb)
    max_rgb = np.array(max_rgb)
    # Threshold color FA
    cc_mask = np.all(
        (in_cfa >= min_rgb[None, :]) & (in_cfa <= max_rgb[None, :]),
        axis=-1,
    )
    # Apply bounding box and WM mask
    cc_mask *= cc_box & mask
    struct = nd.generate_binary_structure(cc_mask.ndim, cc_mask.ndim - 1)
    # Perform a closing followed by opening operations on the FA.
    cc_mask = nd.binary_closing(
        cc_mask,
        structure=struct,
    )
    cc_mask = nd.binary_opening(
        cc_mask,
        structure=struct,
    )
    if clean_mask:
        from dipy.segment.mask import clean_cc_mask

        cc_mask = clean_cc_mask(cc_mask)
    return cc_mask
