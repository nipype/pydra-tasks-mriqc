import attrs
from fileformats.generic import Directory, File
import json
import logging
import numpy as np
from pathlib import Path
from pydra.compose import python, shell, workflow
from .conform_image import ConformImage
from pydra.utils.typing import MultiInputObj
import typing as ty
import yaml


logger = logging.getLogger(__name__)


NUMPY_DTYPE = {
    1: np.uint8,
    2: np.uint8,
    4: np.uint16,
    8: np.uint32,
    64: np.float32,
    256: np.uint8,
    1024: np.uint32,
    1280: np.uint32,
    1536: np.float32,
}

OUT_FILE = "{prefix}_conformed{ext}"
