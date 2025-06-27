from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.number_of_shells import NumberOfShells
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_numberofshells_1():
    task = NumberOfShells()
    task.inputs.in_bvals = File.sample(seed=0)
    task.inputs.b0_threshold = 50
    task.inputs.dsi_threshold = 11
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
