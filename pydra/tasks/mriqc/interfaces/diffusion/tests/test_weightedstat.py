from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.weighted_stat import WeightedStat
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_weightedstat_1():
    task = WeightedStat()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.stat = "mean"
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
