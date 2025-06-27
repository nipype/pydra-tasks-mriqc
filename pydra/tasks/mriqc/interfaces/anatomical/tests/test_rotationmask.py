from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.anatomical.rotation_mask import RotationMask
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_rotationmask_1():
    task = RotationMask()
    task.inputs.in_file = File.sample(seed=0)
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
