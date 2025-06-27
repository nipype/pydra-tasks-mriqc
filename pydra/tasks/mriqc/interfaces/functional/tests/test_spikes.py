from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.functional.spikes import Spikes
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_spikes_1():
    task = Spikes()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.in_mask = File.sample(seed=1)
    task.inputs.invert_mask = False
    task.inputs.no_zscore = False
    task.inputs.detrend = True
    task.inputs.spike_thresh = 6.0
    task.inputs.skip_frames = 0
    task.inputs.out_tsz = "spikes_tsz.txt"
    task.inputs.out_spikes = "spikes_idx.txt"
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
