from fileformats.generic import File
from fileformats.medimage import Nifti1
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.transitional.gcor import GCOR
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_gcor_1():
    task = GCOR()
    task.inputs.in_file = Nifti1.sample(seed=0)
    task.inputs.mask = File.sample(seed=1)
    print(f"CMDLINE: {task.cmdline}\n\n")
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)


@pytest.mark.xfail
def test_gcor_2():
    task = GCOR()
    task.inputs.in_file = Nifti1.sample(seed=0)
    task.inputs.nfirst = 4
    print(f"CMDLINE: {task.cmdline}\n\n")
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
