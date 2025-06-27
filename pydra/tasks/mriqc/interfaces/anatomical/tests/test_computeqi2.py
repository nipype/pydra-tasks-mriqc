from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.anatomical.compute_qi2 import ComputeQI2
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_computeqi2_1():
    task = ComputeQI2()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.air_msk = File.sample(seed=1)
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
