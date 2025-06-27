import logging
from pydra.compose import workflow
from pydra.tasks.mriqc.workflows.anatomical.base import init_brain_tissue_segmentation
import pytest


logger = logging.getLogger(__name__)


def test_init_brain_tissue_segmentation_build():
    workflow = init_brain_tissue_segmentation()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_init_brain_tissue_segmentation_run():
    workflow = init_brain_tissue_segmentation()
    result = workflow(worker="debug")
    print(result.out)
