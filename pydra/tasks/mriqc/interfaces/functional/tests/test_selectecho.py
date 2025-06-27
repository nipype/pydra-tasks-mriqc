from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.functional.select_echo import SelectEcho
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_selectecho_1():
    task = SelectEcho()
    task.inputs.in_files = [File.sample(seed=0)]
    task.inputs.te_reference = 0.03
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
