from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.spiking_voxels_mask import SpikingVoxelsMask
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_spikingvoxelsmask_1():
    task = SpikingVoxelsMask()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.brain_mask = File.sample(seed=1)
    task.inputs.z_threshold = 3.0
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
