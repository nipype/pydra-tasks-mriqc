import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.bids.iqm_file_sink import IQMFileSink
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_iqmfilesink_1():
    task = IQMFileSink()
    task.inputs.dismiss_entities = ["datatype", "part", "echo", "extension", "suffix"]
    task.inputs._outputs = {}
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
