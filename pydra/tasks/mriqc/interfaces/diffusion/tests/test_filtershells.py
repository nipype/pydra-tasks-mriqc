from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.filter_shells import FilterShells
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_filtershells_1():
    task = FilterShells()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.bvec_file = File.sample(seed=2)
    task.inputs.b_threshold = 1100
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
