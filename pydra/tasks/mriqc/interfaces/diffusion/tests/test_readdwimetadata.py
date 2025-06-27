from fileformats.generic import Directory, File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.read_dwi_metadata import ReadDWIMetadata
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_readdwimetadata_1():
    task = ReadDWIMetadata()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.bids_validate = True
    task.inputs.index_db = Directory.sample(seed=3)
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
