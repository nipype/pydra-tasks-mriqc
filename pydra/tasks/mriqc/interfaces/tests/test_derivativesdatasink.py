from fileformats.generic import Directory, File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.derivatives_data_sink import DerivativesDataSink
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_derivativesdatasink_1():
    task = DerivativesDataSink()
    task.inputs.base_directory = Directory.sample(seed=0)
    task.inputs.check_hdr = True
    task.inputs.compress = []
    task.inputs.dismiss_entities = []
    task.inputs.in_file = [File.sample(seed=5)]
    task.inputs.source_file = [File.sample(seed=7)]
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
