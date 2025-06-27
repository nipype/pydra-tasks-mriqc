from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.split_shells import SplitShells
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_splitshells_1():
    task = SplitShells()
    task.inputs.in_file = File.sample(seed=0)
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
