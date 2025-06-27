import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.webapi.upload_iq_ms import UploadIQMs
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_uploadiqms_1():
    task = UploadIQMs()
    task.inputs.strict = False
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
