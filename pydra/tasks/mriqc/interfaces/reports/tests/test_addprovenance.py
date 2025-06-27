from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.reports.add_provenance import AddProvenance
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_addprovenance_1():
    task = AddProvenance()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.air_msk = File.sample(seed=1)
    task.inputs.rot_msk = File.sample(seed=2)
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
