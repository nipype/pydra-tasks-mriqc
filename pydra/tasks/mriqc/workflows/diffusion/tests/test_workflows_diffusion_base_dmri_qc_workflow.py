from fileformats.medimage import Bval, Bvec
import logging
from pydra.compose import workflow
from pydra.tasks.mriqc.workflows.diffusion.base import dmri_qc_workflow
import pytest


logger = logging.getLogger(__name__)


def test_dmri_qc_workflow_build():
    workflow = dmri_qc_workflow()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_dmri_qc_workflow_run():
    workflow = dmri_qc_workflow(bvals=Bval.sample(), bvecs=Bvec.sample())
    result = workflow(worker="debug")
    print(result.out)
