from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.nipype_ports.algorithms.confounds.non_steady_state_detector import (
    NonSteadyStateDetector,
)
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_nonsteadystatedetector_1():
    task = NonSteadyStateDetector()
    task.inputs.in_file = File.sample(seed=0)
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
