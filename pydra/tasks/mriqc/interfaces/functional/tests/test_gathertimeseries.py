from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.functional.gather_timeseries import GatherTimeseries
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_gathertimeseries_1():
    task = GatherTimeseries()
    task.inputs.dvars = File.sample(seed=0)
    task.inputs.fd = File.sample(seed=1)
    task.inputs.mpars = File.sample(seed=2)
    task.inputs.outliers = File.sample(seed=4)
    task.inputs.quality = File.sample(seed=5)
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
