from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.functional.functional_qc import FunctionalQC
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_functionalqc_1():
    task = FunctionalQC()
    task.inputs.in_epi = File.sample(seed=0)
    task.inputs.in_hmc = File.sample(seed=1)
    task.inputs.in_tsnr = File.sample(seed=2)
    task.inputs.in_mask = File.sample(seed=3)
    task.inputs.direction = "all"
    task.inputs.in_fd = File.sample(seed=5)
    task.inputs.fd_thres = 0.2
    task.inputs.in_dvars = File.sample(seed=7)
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
