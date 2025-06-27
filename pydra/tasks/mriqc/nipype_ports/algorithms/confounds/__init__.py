import attrs
from fileformats.generic import Directory, File
import json
import logging
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from pathlib import Path
from pydra.compose import python, shell, workflow
from .compute_dvars import ComputeDVARS
from .framewise_displacement import FramewiseDisplacement
from .non_steady_state_detector import NonSteadyStateDetector
from .tsnr import TSNR
from pydra.utils.typing import MultiInputObj
import typing as ty
import yaml


logger = logging.getLogger(__name__)


IFLOGGER = logging.getLogger("nipype.interface")
