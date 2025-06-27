from fileformats.generic import File
import logging
from nipype2pydra.testing import PassAfterTimeoutWorker
from pydra.tasks.mriqc.nipype_ports.algorithms.confounds.framewise_displacement import (
    FramewiseDisplacement,
)
import pytest


logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_framewisedisplacement_1():
    task = FramewiseDisplacement()
    task.inputs.in_file = File.sample(seed=0)
    task.inputs.radius = 50
    task.inputs.out_file = "fd_power_2012.txt"
    task.inputs.out_figure = "fd_power_2012.pdf"
    task.inputs.save_plot = False
    task.inputs.normalize = False
    task.inputs.figdpi = 100
    task.inputs.figsize = [11.7, 2.3]
    res = task(worker=PassAfterTimeoutWorker)
    print("RESULT: ", res)
