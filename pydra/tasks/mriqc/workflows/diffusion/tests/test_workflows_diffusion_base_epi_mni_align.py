import logging
from pydra.compose import workflow
from pydra.tasks.mriqc.workflows.diffusion.base import epi_mni_align
import pytest


logger = logging.getLogger(__name__)


def test_epi_mni_align_build():
    workflow = epi_mni_align()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_epi_mni_align_run():
    workflow = epi_mni_align()
    result = workflow(worker="debug")
    print(result.out)
