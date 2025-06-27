from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.extract_orientations import (
    ExtractOrientations,
)
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_extractorientations_1():
    task = ExtractOrientations()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.in_bvec_file = File.sample(seed=2)
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
