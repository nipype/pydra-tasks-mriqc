import attrs
import logging
from pathlib import Path
from pydra.compose import python, workflow
from pydra.tasks.mriqc.workflows.diffusion.output import init_dwi_report_wf
from pydra.tasks.nireports.interfaces.dmri import DWIHeatmap
from pydra.tasks.nireports.interfaces.reporting.base import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)
import typing as ty


logger = logging.getLogger(__name__)


@workflow.define(
    outputs=[
        "snr_report",
        "noise_report",
        "fa_report",
        "md_report",
        "heatmap_report",
        "spikes_report",
        "carpet_report",
        "bmask_report",
    ]
)
def init_dwi_report_wf(
    brain_mask: ty.Any = attrs.NOTHING,
    exec_verbose_reports=False,
    exec_work_dir=None,
    in_avgmap: ty.Any = attrs.NOTHING,
    in_bdict: ty.Any = attrs.NOTHING,
    in_epi: ty.Any = attrs.NOTHING,
    in_fa: ty.Any = attrs.NOTHING,
    in_md: ty.Any = attrs.NOTHING,
    in_parcellation: ty.Any = attrs.NOTHING,
    in_stdmap: ty.Any = attrs.NOTHING,
    name="dwi_report_wf",
    noise_floor: ty.Any = attrs.NOTHING,
    wf_biggest_file_gb=1,
    wf_fd_thres=0.2,
    wf_species="human",
) -> tuple[ty.Any, ty.Any, ty.Any, ty.Any, ty.Any, ty.Any, ty.Any, ty.Any]:
    """
    Write out individual reportlets.

    .. workflow::

        from mriqc.workflows.diffusion.output import init_dwi_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_dwi_report_wf()

    """
    from pydra.tasks.nireports.interfaces import PlotMosaic

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    outputs_ = {
        "snr_report": attrs.NOTHING,
        "noise_report": attrs.NOTHING,
        "fa_report": attrs.NOTHING,
        "md_report": attrs.NOTHING,
        "heatmap_report": attrs.NOTHING,
        "spikes_report": attrs.NOTHING,
        "carpet_report": attrs.NOTHING,
        "bmask_report": attrs.NOTHING,
    }

    # from mriqc.interfaces.reports import IndividualReport
    verbose = exec_verbose_reports
    mem_gb = wf_biggest_file_gb
    reportlets_dir = exec_work_dir / "reportlets"

    # Set FD threshold
    # inputnode.inputs.fd_thres = wf_fd_thres
    mosaic_fa = workflow.add(
        PlotMosaic(cmap="Greys_r", bbox_mask_file=brain_mask, in_file=in_fa),
        name="mosaic_fa",
    )
    mosaic_md = workflow.add(
        PlotMosaic(cmap="Greys_r", bbox_mask_file=brain_mask, in_file=in_md),
        name="mosaic_md",
    )
    mosaic_snr = workflow.add(
        SimpleBeforeAfter(
            after_label="Standard Deviation",
            before_label="Average",
            dismiss_affine=True,
            fixed_params={"cmap": "viridis"},
            moving_params={"cmap": "Greys_r"},
            after=in_stdmap,
            before=in_avgmap,
            wm_seg=brain_mask,
        ),
        name="mosaic_snr",
    )
    mosaic_noise = workflow.add(
        PlotMosaic(cmap="viridis_r", only_noise=True, in_file=in_avgmap),
        name="mosaic_noise",
    )
    if wf_species.lower() in ("rat", "mouse"):
        mosaic_noise.inputs.inputs.view = ["coronal", "axial"]
        mosaic_fa.inputs.inputs.view = ["coronal", "axial"]
        mosaic_md.inputs.inputs.view = ["coronal", "axial"]

    def _gen_entity(inlist):
        return ["00000"] + [f"{int(round(bval, 0)):05d}" for bval in inlist]

    # fmt: off


    outputs_['snr_report'] = mosaic_snr.out_report
    outputs_['noise_report'] = mosaic_noise.out_file
    outputs_['fa_report'] = mosaic_fa.out_file
    outputs_['md_report'] = mosaic_md.out_file
    # fmt: on
    get_wm = workflow.add(
        python.define(_get_wm)(in_file=in_parcellation), name="get_wm"
    )
    plot_heatmap = workflow.add(
        DWIHeatmap(
            scalarmap_label="Shell-wise Fractional Anisotropy (FA)",
            b_indices=in_bdict,
            in_file=in_epi,
            mask_file=get_wm.out,
            scalarmap=in_fa,
            sigma=noise_floor,
        ),
        name="plot_heatmap",
    )

    # fmt: off
    outputs_['heatmap_report'] = plot_heatmap.out_file
    # fmt: on

    return tuple(outputs_)


def _get_wm(in_file, radius=2):

    from pathlib import Path
    import nibabel as nb
    import numpy as np
    from pydra.tasks.mriqc.nipype_ports.utils.filemanip import fname_presuffix
    from scipy import ndimage as ndi
    from skimage.morphology import ball

    parc = nb.load(in_file)
    hdr = parc.header.copy()
    data = np.array(parc.dataobj, dtype=hdr.get_data_dtype())
    wm_mask = ndi.binary_erosion((data == 1) | (data == 2), ball(radius))
    hdr.set_data_dtype(np.uint8)
    out_wm = fname_presuffix(in_file, suffix="wm", newpath=str(Path.cwd()))
    parc.__class__(
        wm_mask.astype(np.uint8),
        parc.affine,
        hdr,
    ).to_filename(out_wm)
    return out_wm
