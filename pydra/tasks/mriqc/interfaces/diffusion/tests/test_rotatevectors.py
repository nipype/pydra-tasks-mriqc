from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.rotate_vectors import RotateVectors
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_rotatevectors_1():
    task = RotateVectors()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.reference = File.sample(seed=1)
    task.inputs.transforms = File.sample(seed=2)
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
