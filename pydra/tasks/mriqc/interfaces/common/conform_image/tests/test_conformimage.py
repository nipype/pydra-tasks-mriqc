from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.common.conform_image.conform_image import ConformImage
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_conformimage_1():
    task = ConformImage()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.check_ras = True
    task.inputs.check_dtype = True
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
