from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.common.ensure_size.ensure_size import EnsureSize
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_ensuresize_1():
    task = EnsureSize()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.in_mask = File.sample(seed=1)
    task.inputs.pixel_size = 2.0
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
