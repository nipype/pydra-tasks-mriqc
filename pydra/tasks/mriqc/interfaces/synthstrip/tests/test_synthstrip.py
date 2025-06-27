from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.synthstrip.synth_strip import SynthStrip
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_synthstrip_1():
    task = SynthStrip()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.use_gpu = False
    task.inputs.model = File.sample(seed=2)
    task.inputs.border_mm = 1
    print(f"CMDLINE: {task.cmdline}\n\n")
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
