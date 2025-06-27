from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.cc_segmentation import CCSegmentation
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_ccsegmentation_1():
    task = CCSegmentation()
    task.inputs.in_fa = File.sample(seed=0)
    task.inputs.in_cfa = File.sample(seed=1)
    task.inputs.min_rgb = [0.4, 0.008, 0.008]
    task.inputs.max_rgb = [1.1, 0.25, 0.25]
    task.inputs.wm_threshold = 0.35
    task.inputs.clean_mask = False
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
