import attrs
from fileformats.generic import Directory, File
import json
import logging
import numpy as np
from pathlib import Path
from pydra.compose import python, shell, workflow
from .artifact_mask import ArtifactMask
from .compute_qi2 import ComputeQI2
from .harmonize import Harmonize
from .rotation_mask import RotationMask
from .structural_qc import StructuralQC
from pydra.utils.typing import MultiInputObj
import scipy.ndimage as nd
import typing as ty
import yaml


logger = logging.getLogger(__name__)


def artifact_mask(imdata, airdata, distance, zscore=10.0):
    """Compute a mask of artifacts found in the air region."""
    from statsmodels.robust.scale import mad

    qi1_msk = np.zeros(imdata.shape, dtype=bool)
    bg_data = imdata[airdata]
    if (bg_data > 0).sum() < 10:
        return qi1_msk
    # Standardize the distribution of the background
    bg_spread = mad(bg_data[bg_data > 0])
    bg_data[bg_data > 0] = bg_data[bg_data > 0] / bg_spread
    # Apply this threshold to the background voxels to identify voxels
    # contributing artifacts.
    qi1_msk[airdata] = bg_data > zscore
    qi1_msk[distance < 0.10] = False
    # Create a structural element to be used in an opening operation.
    struct = nd.generate_binary_structure(3, 1)
    qi1_msk = nd.binary_opening(qi1_msk, struct).astype(np.uint8)
    return qi1_msk


def fuzzy_jaccard(in_tpms, in_mni_tpms):

    overlaps = []
    for tpm, mni_tpm in zip(in_tpms, in_mni_tpms):
        tpm = tpm.reshape(-1)
        mni_tpm = mni_tpm.reshape(-1)
        num = np.min([tpm, mni_tpm], axis=0).sum()
        den = np.max([tpm, mni_tpm], axis=0).sum()
        overlaps.append(float(num / den))
    return overlaps
