from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.piesno import PIESNO
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_piesno_1():
    task = PIESNO()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.n_channels = 4
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
