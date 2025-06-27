import logging
from math import sqrt
import numpy as np
import os.path as op
from scipy.stats import kurtosis


logger = logging.getLogger(__name__)


def art_qi1(airmask, artmask):
    r"""
    Detect artifacts in the image using the method described in [Mortamet2009]_.
    Calculates :math:`\text{QI}_1`, as the proportion of voxels with intensity
    corrupted by artifacts normalized by the number of voxels in the "*hat*"
    mask (i.e., the background region above the nasio-occipital plane):

    .. math ::

        \text{QI}_1 = \frac{1}{N} \sum\limits_{x\in X_\text{art}} 1

    Near-zero values are better.
    If :math:`\text{QI}_1 = -1`, then the "*hat*" mask (background) was empty
    and the dataset is likely a skull-stripped image or has been heavily
    post-processed.

    :param numpy.ndarray airmask: input air mask, without artifacts
    :param numpy.ndarray artmask: input artifacts mask

    """
    if airmask.sum() < 1:
        return -1.0
    # Count the ratio between artifacts and the total voxels in "hat" mask
    return float(artmask.sum() / (airmask.sum() + artmask.sum()))


def art_qi2(
    img,
    airmask,
    min_voxels=int(1e3),
    max_voxels=int(3e5),
    save_plot=True,
    coil_elements=32,
):
    r"""
    Calculates :math:`\text{QI}_2`, based on the goodness-of-fit of a centered
    :math:`\chi^2` distribution onto the intensity distribution of
    non-artifactual background (within the "hat" mask):


    .. math ::

        \chi^2_n = \frac{2}{(\sigma \sqrt{2})^{2n} \, (n - 1)!}x^{2n - 1}\, e^{-\frac{x}{2}}

    where :math:`n` is the number of coil elements.

    :param numpy.ndarray img: input data
    :param numpy.ndarray airmask: input air mask without artifacts

    """
    from pydra.tasks.nireports.reportlets.nuisance import plot_qi2
    from scipy.stats import chi2
    from sklearn.neighbors import KernelDensity

    # S. Ogawa was born
    np.random.seed(1191935)
    data = np.nan_to_num(img[airmask > 0], posinf=0.0)
    data[data < 0] = 0
    # Write out figure of the fitting
    out_file = op.abspath("error.svg")
    with open(out_file, "w") as ofh:
        ofh.write("<p>Background noise fitting could not be plotted.</p>")
    if (data > 0).sum() < min_voxels:
        return 0.0, out_file
    data *= 100 / np.percentile(data, 99)
    modelx = data if len(data) < max_voxels else np.random.choice(data, size=max_voxels)
    x_grid = np.linspace(0.0, 110, 1000)
    # Estimate data pdf with KDE on a random subsample
    kde_skl = KernelDensity(kernel="gaussian", bandwidth=4.0).fit(modelx[:, np.newaxis])
    kde = np.exp(kde_skl.score_samples(x_grid[:, np.newaxis]))
    # Find cutoff
    kdethi = np.argmax(kde[::-1] > kde.max() * 0.5)
    # Fit X^2
    param = chi2.fit(modelx, coil_elements)
    chi_pdf = chi2.pdf(x_grid, *param[:-2], loc=param[-2], scale=param[-1])
    # Compute goodness-of-fit (gof)
    gof = float(np.abs(kde[-kdethi:] - chi_pdf[-kdethi:]).mean())
    if save_plot:
        out_file = plot_qi2(x_grid, kde, chi_pdf, modelx, kdethi)
    return gof, out_file


def cjv(mu_wm, mu_gm, sigma_wm, sigma_gm):
    r"""
    Calculate the :abbr:`CJV (coefficient of joint variation)`, a measure
    related to :abbr:`SNR (Signal-to-Noise Ratio)` and
    :abbr:`CNR (Contrast-to-Noise Ratio)` that is presented as a proxy for
    the :abbr:`INU (intensity non-uniformity)` artifact [Ganzetti2016]_.
    Lower is better.

    .. math::

        \text{CJV} = \frac{\sigma_\text{WM} + \sigma_\text{GM}}{|\mu_\text{WM} - \mu_\text{GM}|}.

    :param float mu_wm: mean of signal within white-matter mask.
    :param float mu_gm: mean of signal within gray-matter mask.
    :param float sigma_wm: standard deviation of signal within white-matter mask.
    :param float sigma_gm: standard deviation of signal within gray-matter mask.

    :return: the computed CJV


    """
    return float((sigma_wm + sigma_gm) / abs(mu_wm - mu_gm))


def cnr(mu_wm, mu_gm, sigma_air, sigma_wm, sigma_gm):
    r"""
    Calculate the :abbr:`CNR (Contrast-to-Noise Ratio)` [Magnota2006]_.
    Higher values are better.

    .. math::

        \text{CNR} = \frac{|\mu_\text{GM} - \mu_\text{WM} |}{\sqrt{\sigma_B^2 +
        \sigma_\text{WM}^2 + \sigma_\text{GM}^2}},

    where :math:`\sigma_B` is the standard deviation of the noise distribution within
    the air (background) mask.


    :param float mu_wm: mean of signal within white-matter mask.
    :param float mu_gm: mean of signal within gray-matter mask.
    :param float sigma_air: standard deviation of the air surrounding the head ("hat" mask).
    :param float sigma_wm: standard deviation within white-matter mask.
    :param float sigma_gm: standard within gray-matter mask.

    :return: the computed CNR

    """
    return float(abs(mu_wm - mu_gm) / sqrt(sigma_air**2 + sigma_gm**2 + sigma_wm**2))


def efc(img, framemask=None, decimals=4):
    r"""
    Calculate the :abbr:`EFC (Entropy Focus Criterion)` [Atkinson1997]_.
    Uses the Shannon entropy of voxel intensities as an indication of ghosting
    and blurring induced by head motion. A range of low values is better,
    with EFC = 0 for all the energy concentrated in one pixel.

    .. math::

        \text{E} = - \sum_{j=1}^N \frac{x_j}{x_\text{max}}
        \ln \left[\frac{x_j}{x_\text{max}}\right]

    with :math:`x_\text{max} = \sqrt{\sum_{j=1}^N x^2_j}`.

    The original equation is normalized by the maximum entropy, so that the
    :abbr:`EFC (Entropy Focus Criterion)` can be compared across images with
    different dimensions:

    .. math::

        \text{EFC} = \left( \frac{N}{\sqrt{N}} \, \log{\sqrt{N}^{-1}} \right) \text{E}

    :param numpy.ndarray img: input data
    :param numpy.ndarray framemask: a mask of empty voxels inserted after a rotation of
      data

    """
    if framemask is None:
        framemask = np.zeros_like(img, dtype=np.uint8)
    n_vox = np.sum(1 - framemask)
    # Calculate the maximum value of the EFC (which occurs any time all
    # voxels have the same value)
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * np.log(1.0 / np.sqrt(n_vox))
    # Calculate the total image energy
    b_max = np.sqrt((img[framemask == 0] ** 2).sum())
    # Calculate EFC (add 1e-16 to the image data to keep log happy)
    return round(
        float(
            (1.0 / efc_max)
            * np.sum(
                (img[framemask == 0] / b_max)
                * np.log((img[framemask == 0] + 1e-16) / b_max)
            ),
        ),
        decimals,
    )


def fber(img, headmask, rotmask=None, decimals=4):
    r"""
    Calculate the :abbr:`FBER (Foreground-Background Energy Ratio)` [Shehzad2015]_,
    defined as the mean energy of image values within the head relative
    to outside the head.
    Higher values are better, and an FBER=-1.0 indicates that there is no signal
    outside the head mask (e.g., a skull-stripped dataset).

    .. math::

        \text{FBER} = \frac{E[|F|^2]}{E[|B|^2]}


    :param numpy.ndarray img: input data
    :param numpy.ndarray headmask: a mask of the head (including skull, skin, etc.)
    :param numpy.ndarray rotmask: a mask of empty voxels inserted after a rotation of
      data

    """
    fg_mu = np.median(np.abs(img[headmask > 0]) ** 2)
    airmask = np.ones_like(headmask, dtype=np.uint8)
    airmask[headmask > 0] = 0
    if rotmask is not None:
        airmask[rotmask > 0] = 0
    bg_mu = np.median(np.abs(img[airmask == 1]) ** 2)
    if bg_mu < 1.0e-3:
        return -1.0
    return round(float(fg_mu / bg_mu), decimals)


def rpve(pvms, seg):
    """
    Computes the :abbr:`rPVe (residual partial voluming error)`
    of each tissue class.

    .. math ::

        \\text{rPVE}^k = \\frac{1}{N} \\left[ \\sum\\limits_{p^k_i  \\in [0.5, P_{98}]} p^k_i + \\sum\\limits_{p^k_i \\in [P_{2}, 0.5)} 1 - p^k_i \\right]

    """
    pvfs = {}
    for k, lid in list(FSL_FAST_LABELS.items()):
        if lid == 0:
            continue
        pvmap = pvms[lid - 1]
        pvmap[pvmap < 0.0] = 0.0
        pvmap[pvmap >= 1.0] = 1.0
        totalvol = np.sum(pvmap > 0.0)
        upth = np.percentile(pvmap[pvmap > 0], 98)
        loth = np.percentile(pvmap[pvmap > 0], 2)
        pvmap[pvmap < loth] = 0
        pvmap[pvmap > upth] = 0
        pvfs[k] = (
            pvmap[pvmap > 0.5].sum() + (1.0 - pvmap[pvmap <= 0.5]).sum()
        ) / totalvol
    return {k: float(v) for k, v in list(pvfs.items())}


def snr(mu_fg, sigma_fg, n):
    r"""
    Calculate the :abbr:`SNR (Signal-to-Noise Ratio)`.
    The estimation may be provided with only one foreground region in
    which the noise is computed as follows:

    .. math::

        \text{SNR} = \frac{\mu_F}{\sigma_F\sqrt{n/(n-1)}},

    where :math:`\mu_F` is the mean intensity of the foreground and
    :math:`\sigma_F` is the standard deviation of the same region.

    :param float mu_fg: mean of foreground.
    :param float sigma_fg: standard deviation of foreground.
    :param int n: number of voxels in foreground mask.

    :return: the computed SNR

    """
    return float(mu_fg / (sigma_fg * sqrt(n / (n - 1))))


def snr_dietrich(mu_fg, mad_air=0.0, sigma_air=1.0):
    r"""
    Calculate the :abbr:`SNR (Signal-to-Noise Ratio)`.

    This must be an air mask around the head, and it should not contain artifacts.
    The computation is done following the eq. A.12 of [Dietrich2007]_, which
    includes a correction factor in the estimation of the standard deviation of
    air and its Rayleigh distribution:

    .. math::

        \text{SNR} = \frac{\mu_F}{\sqrt{\frac{2}{4-\pi}}\,\sigma_\text{air}}.


    :param float mu_fg: mean of foreground.
    :param float sigma_air: standard deviation of the air surrounding the head ("hat" mask).

    :return: the computed SNR for the foreground segmentation

    """
    if mad_air > 1.0:
        return float(DIETRICH_FACTOR * mu_fg / mad_air)
    logger.warning(
        "Estimated signal variation in the background was too small "
        f"(MAD={mad_air}, sigma={sigma_air})",
    )
    return float(DIETRICH_FACTOR * mu_fg / sigma_air) if sigma_air > 1e-3 else -1.0


def summary_stats(
    data: np.ndarray,
    pvms: dict[str, np.ndarray],
    rprec_data: int = 0,
    rprec_prob: int = 3,
    decimals: int = 4,
) -> dict[str, dict[str, float]]:
    """
    Estimates weighted summary statistics for each tissue distribution in the data.

    This function calculates the mean, median, standard deviation, kurtosis, median
    absolute deviation (MAD), the 95th and 5th percentiles, and the number of voxels for
    each tissue distribution defined by a label in the provided partial volume maps (pvms).

    Parameters
    ----------
    data : :obj:`~numpy.ndarray` (float, 3D)
        A three-dimensional array of data from which summary statistics will be extracted.
    pvms : :obj:`dict` of :obj:`str` keys and :obj:`~numpy.ndarray` (float, 3D) values
        A dictionary of partial volume maps where the key indicates the label of a
        region-of-interest (ROI) and the values are three-dimensional arrays matched in size
        with `data` and containing the probability/fraction of the voxel containing the given
        label.
    rprec_data : :obj:`int`, optional (default=0)
        Number of decimal places to round the data array before calculation. Rounding
        alleviates floating-point error variability by explicitly rounding before
        quantification operations.
    rprec_prob : :obj:`int`, optional (default=3)
        Number of decimal places to round the probability maps before calculation. Rounding
        alleviates floating-point error variability by explicitly rounding before
        quantification operations.

    Returns
    -------
    :obj:`dict`
        A dictionary where the keys are labels from the ``pvms`` dictionary and the values
        are dictionaries containing the following keys for each tissue distribution:

            * ``'mean'``: :obj:`float` - Mean value
            * ``'median'``: :obj:`float` - Median value
            * ``'p95'``: :obj:`float` - 95th percentile
            * ``'p05'``: :obj:`float` - 5th percentile
            * ``'k'``: :obj:`float` - Kurtosis
            * ``'stdv'``: :obj:`float` - Standard deviation
            * ``'mad'``: :obj:`float` - Median absolute deviation
            * ``'n'``: :obj:`int` - Number of voxels in the tissue distribution

    """
    from statsmodels.robust.scale import mad
    from statsmodels.stats.weightstats import DescrStatsW

    output = {}
    for label, probmap in pvms.items():
        wstats = DescrStatsW(
            data=np.round(data.reshape(-1), rprec_data),
            weights=np.round(probmap.astype(np.float32).reshape(-1), rprec_prob),
        )
        nvox = probmap.sum()
        p05, median, p95 = wstats.quantile(
            np.array([0.05, 0.50, 0.95]), return_pandas=False
        )
        thresholded = data[probmap > (0.5 * probmap.max())]
        output[label] = {
            "mean": round(float(wstats.mean), decimals),
            "median": round(float(median), decimals),
            "p95": round(float(p95), decimals),
            "p05": round(float(p05), decimals),
            "k": round(float(kurtosis(thresholded)), decimals),
            "stdv": round(float(wstats.std), decimals),
            "mad": round(float(mad(thresholded, center=median)), decimals),
            "n": float(nvox),
        }
    return output


def volume_fraction(pvms):
    r"""
    Computes the :abbr:`ICV (intracranial volume)` fractions
    corresponding to the (partial volume maps).

    .. math ::

        \text{ICV}^k = \frac{\sum_i p^k_i}{\sum\limits_{x \in X_\text{brain}} 1}

    :param list pvms: list of :code:`numpy.ndarray` of partial volume maps.

    """
    tissue_vfs = {}
    total = 0
    for k, lid in list(FSL_FAST_LABELS.items()):
        if lid == 0:
            continue
        tissue_vfs[k] = pvms[lid - 1].sum()
        total += tissue_vfs[k]
    for k in list(tissue_vfs.keys()):
        tissue_vfs[k] /= total
    return {k: float(v) for k, v in list(tissue_vfs.items())}


def wm2max(img, mu_wm):
    r"""
    Calculate the :abbr:`WM2MAX (white-matter-to-max ratio)`,
    defined as the maximum intensity found in the volume w.r.t. the
    mean value of the white matter tissue. Values close to 1.0 are
    better:

    .. math ::

        \text{WM2MAX} = \frac{\mu_\text{WM}}{P_{99.95}(X)}

    """
    return float(mu_wm / np.percentile(img.reshape(-1), 99.95))


DIETRICH_FACTOR = 0.6551364  # 1.0 / sqrt(2 / (4 - pi))

FSL_FAST_LABELS = {"csf": 1, "gm": 2, "wm": 3, "bg": 0}
