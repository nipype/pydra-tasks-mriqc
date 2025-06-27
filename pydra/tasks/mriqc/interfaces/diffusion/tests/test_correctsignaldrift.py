from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.correct_signal_drift import (
    CorrectSignalDrift,
)
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_correctsignaldrift_1():
    task = CorrectSignalDrift()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.bias_file = File.sample(seed=1)
    task.inputs.brainmask_file = File.sample(seed=2)
    task.inputs.bval_file = File.sample(seed=4)
    task.inputs.full_epi = File.sample(seed=5)
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
