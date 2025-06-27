from fileformats.medimage import NiftiGzX, T1Weighted
import logging
from pydra.compose import workflow
from pydra.tasks.mriqc.workflows.anatomical.base import anat_qc_workflow
import pytest


logger = logging.getLogger(__name__)


def test_anat_qc_workflow_build():
    workflow = anat_qc_workflow()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_anat_qc_workflow_run():
    workflow = anat_qc_workflow(in_file=NiftiGzX[T1Weighted].sample())
    result = workflow(worker="debug")
    print(result.out)
