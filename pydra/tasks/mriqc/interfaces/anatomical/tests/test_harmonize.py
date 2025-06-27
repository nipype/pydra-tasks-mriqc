from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.anatomical.harmonize import Harmonize
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_harmonize_1():
    task = Harmonize()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.wm_mask = File.sample(seed=1)
    task.inputs.erodemsk = True
    task.inputs.thresh = 0.9
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
