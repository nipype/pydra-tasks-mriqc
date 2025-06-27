import attrs
import logging
from pathlib import Path
from pydra.compose import workflow
import typing as ty


logger = logging.getLogger(__name__)


@workflow.define(
    outputs=[
        "zoom_report",
        "bg_report",
        "segm_report",
        "bmask_report",
        "artmask_report",
        "airmask_report",
        "headmask_report",
    ]
)
def init_anat_report_wf(
    airmask: ty.Any = attrs.NOTHING,
    artmask: ty.Any = attrs.NOTHING,
    brainmask: ty.Any = attrs.NOTHING,
    exec_verbose_reports=False,
    exec_work_dir=None,
    headmask: ty.Any = attrs.NOTHING,
    in_ras: ty.Any = attrs.NOTHING,
    name: str = "anat_report_wf",
    segmentation: ty.Any = attrs.NOTHING,
    wf_species="human",
) -> ["ty.Any", "ty.Any", "ty.Any", "ty.Any", "ty.Any", "ty.Any", "ty.Any"]:
    """
    Generate the components of the individual report.

    .. workflow::

        from mriqc.workflows.anatomical.output import init_anat_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_anat_report_wf()

    """
    from pydra.tasks.nireports.interfaces import PlotMosaic

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    outputs_ = {
        "zoom_report": attrs.NOTHING,
        "bg_report": attrs.NOTHING,
        "segm_report": attrs.NOTHING,
        "bmask_report": attrs.NOTHING,
        "artmask_report": attrs.NOTHING,
        "airmask_report": attrs.NOTHING,
        "headmask_report": attrs.NOTHING,
    }

    # from mriqc.interfaces.reports import IndividualReport
    verbose = exec_verbose_reports
    reportlets_dir = exec_work_dir / "reportlets"

    mosaic_zoom = workflow.add(
        PlotMosaic(cmap="Greys_r", bbox_mask_file=brainmask, in_file=in_ras),
        name="mosaic_zoom",
    )
    mosaic_noise = workflow.add(
        PlotMosaic(cmap="viridis_r", only_noise=True, in_file=in_ras),
        name="mosaic_noise",
    )
    if wf_species.lower() in ("rat", "mouse"):
        mosaic_zoom.inputs.inputs.view = ["coronal", "axial"]
        mosaic_noise.inputs.inputs.view = ["coronal", "axial"]

    # fmt: off
    outputs_['zoom_report'] = mosaic_zoom.out_file
    outputs_['bg_report'] = mosaic_noise.out_file
    # fmt: on

    from pydra.tasks.nireports.interfaces import PlotContours

    display_mode = "y" if wf_species.lower() in ("rat", "mouse") else "z"
    plot_segm = workflow.add(
        PlotContours(
            colors=["r", "g", "b"],
            cut_coords=10,
            display_mode=display_mode,
            levels=[0.5, 1.5, 2.5],
            in_contours=segmentation,
            in_file=in_ras,
        ),
        name="plot_segm",
    )

    plot_bmask = workflow.add(
        PlotContours(
            colors=["r"],
            cut_coords=10,
            display_mode=display_mode,
            levels=[0.5],
            out_file="bmask",
            in_contours=brainmask,
            in_file=in_ras,
        ),
        name="plot_bmask",
    )

    plot_artmask = workflow.add(
        PlotContours(
            colors=["r"],
            cut_coords=10,
            display_mode=display_mode,
            levels=[0.5],
            out_file="artmask",
            saturate=True,
            in_contours=artmask,
            in_file=in_ras,
        ),
        name="plot_artmask",
    )

    # NOTE: humans switch on these two to coronal view.
    display_mode = "y" if wf_species.lower() in ("rat", "mouse") else "x"
    plot_airmask = workflow.add(
        PlotContours(
            colors=["r"],
            cut_coords=6,
            display_mode=display_mode,
            levels=[0.5],
            out_file="airmask",
            in_contours=airmask,
            in_file=in_ras,
        ),
        name="plot_airmask",
    )

    plot_headmask = workflow.add(
        PlotContours(
            colors=["r"],
            cut_coords=6,
            display_mode=display_mode,
            levels=[0.5],
            out_file="headmask",
            in_contours=headmask,
            in_file=in_ras,
        ),
        name="plot_headmask",
    )

    # fmt: off
    outputs_['bmask_report'] = plot_bmask.out_file
    outputs_['segm_report'] = plot_segm.out_file
    outputs_['artmask_report'] = plot_artmask.out_file
    outputs_['headmask_report'] = plot_headmask.out_file
    outputs_['airmask_report'] = plot_airmask.out_file
    # fmt: on

    return tuple(outputs_)
