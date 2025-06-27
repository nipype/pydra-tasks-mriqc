from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.interfaces.diffusion.diffusion_qc import DiffusionQC
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_diffusionqc_1():
    task = DiffusionQC()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.in_b0 = File.sample(seed=1)
    task.inputs.in_shells = [File.sample(seed=2)]
    task.inputs.in_bval_file = File.sample(seed=4)
    task.inputs.in_fa = File.sample(seed=8)
    task.inputs.in_fa_nans = File.sample(seed=9)
    task.inputs.in_fa_degenerate = File.sample(seed=10)
    task.inputs.in_cfa = File.sample(seed=11)
    task.inputs.in_md = File.sample(seed=12)
    task.inputs.brain_mask = File.sample(seed=13)
    task.inputs.wm_mask = File.sample(seed=14)
    task.inputs.cc_mask = File.sample(seed=15)
    task.inputs.spikes_mask = File.sample(seed=16)
    task.inputs.direction = "all"
    task.inputs.in_fd = File.sample(seed=19)
    task.inputs.fd_thres = 0.2
    task.inputs.piesno_sigma = -1.0
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
