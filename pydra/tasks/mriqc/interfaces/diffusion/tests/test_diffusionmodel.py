from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.diffusion_model import DiffusionModel
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_diffusionmodel_1():
    task = DiffusionModel()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.bvec_file = File.sample(seed=2)
    task.inputs.brain_mask = File.sample(seed=3)
    task.inputs.decimals = 3
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
