import attrs
import logging
from pathlib import Path
from pydra.compose import python, workflow
import typing as ty


logger = logging.getLogger(__name__)


@workflow.define(
    outputs=[
        "mean_report",
        "stdev_report",
        "background_report",
        "zoomed_report",
        "carpet_report",
        "spikes_report",
    ]
)
def init_func_report_wf(
    brainmask: ty.Any = attrs.NOTHING,
    epi_mean: ty.Any = attrs.NOTHING,
    epi_parc: ty.Any = attrs.NOTHING,
    exec_verbose_reports=False,
    exec_work_dir=None,
    fd_thres: ty.Any = attrs.NOTHING,
    hmc_epi: ty.Any = attrs.NOTHING,
    hmc_fd: ty.Any = attrs.NOTHING,
    in_dvars: ty.Any = attrs.NOTHING,
    in_fft: ty.Any = attrs.NOTHING,
    in_ras: ty.Any = attrs.NOTHING,
    in_spikes: ty.Any = attrs.NOTHING,
    in_stddev: ty.Any = attrs.NOTHING,
    meta_sidecar: ty.Any = attrs.NOTHING,
    name="func_report_wf",
    outliers: ty.Any = attrs.NOTHING,
    wf_biggest_file_gb=1,
    wf_fft_spikes_detector=False,
    wf_species="human",
) -> tuple[ty.Any, ty.Any, ty.Any, ty.Any, ty.Any, ty.Any]:
    """
    Write out individual reportlets.

    .. workflow::

        from mriqc.workflows.functional.output import init_func_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_func_report_wf()

    """
    from pydra.tasks.nireports.interfaces import FMRISummary, PlotMosaic, PlotSpikes

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    outputs_ = {
        "mean_report": attrs.NOTHING,
        "stdev_report": attrs.NOTHING,
        "background_report": attrs.NOTHING,
        "zoomed_report": attrs.NOTHING,
        "carpet_report": attrs.NOTHING,
        "spikes_report": attrs.NOTHING,
    }

    from pydra.tasks.niworkflows.interfaces.morphology import (
        BinaryDilation,
        BinarySubtraction,
    )
    from pydra.tasks.mriqc.interfaces.functional import Spikes

    # from mriqc.interfaces.reports import IndividualReport
    verbose = exec_verbose_reports
    mem_gb = wf_biggest_file_gb["bold"]
    reportlets_dir = exec_work_dir / "reportlets"

    # Set FD threshold

    spmask = workflow.add(
        python.define(spikes_mask, outputs=["out_file", "out_plot"])(in_file=in_ras),
        name="spmask",
    )
    spikes_bg = workflow.add(
        Spikes(detrend=False, no_zscore=True, in_file=in_ras, in_mask=spmask.out_file),
        name="spikes_bg",
    )
    # Generate crown mask
    # Create the crown mask
    dilated_mask = workflow.add(BinaryDilation(in_mask=brainmask), name="dilated_mask")
    subtract_mask = workflow.add(
        BinarySubtraction(in_base=dilated_mask.out_mask, in_subtract=brainmask),
        name="subtract_mask",
    )
    parcels = workflow.add(
        python.define(_carpet_parcellation)(
            crown_mask=subtract_mask.out_mask, segmentation=epi_parc
        ),
        name="parcels",
    )
    bigplot = workflow.add(
        FMRISummary(
            dvars=in_dvars,
            fd=hmc_fd,
            fd_thres=fd_thres,
            in_func=hmc_epi,
            in_segm=parcels.out,
            in_spikes_bg=spikes_bg.out_tsz,
            outliers=outliers,
            tr=meta_sidecar,
        ),
        name="bigplot",
    )
    # fmt: off
    bigplot.inputs.tr = meta_sidecar
    # fmt: on
    mosaic_mean = workflow.add(
        PlotMosaic(
            cmap="Greys_r", out_file="plot_func_mean_mosaic1.svg", in_file=epi_mean
        ),
        name="mosaic_mean",
    )
    mosaic_stddev = workflow.add(
        PlotMosaic(
            cmap="viridis",
            out_file="plot_func_stddev_mosaic2_stddev.svg",
            in_file=in_stddev,
        ),
        name="mosaic_stddev",
    )
    mosaic_zoom = workflow.add(
        PlotMosaic(cmap="Greys_r", bbox_mask_file=brainmask, in_file=epi_mean),
        name="mosaic_zoom",
    )
    mosaic_noise = workflow.add(
        PlotMosaic(cmap="viridis_r", only_noise=True, in_file=epi_mean),
        name="mosaic_noise",
    )
    if wf_species.lower() in ("rat", "mouse"):
        mosaic_mean.inputs.inputs.view = ["coronal", "axial"]
        mosaic_stddev.inputs.inputs.view = ["coronal", "axial"]
        mosaic_zoom.inputs.inputs.view = ["coronal", "axial"]
        mosaic_noise.inputs.inputs.view = ["coronal", "axial"]

    # fmt: off
    outputs_['mean_report'] = mosaic_mean.out_file
    outputs_['stdev_report'] = mosaic_stddev.out_file
    outputs_['background_report'] = mosaic_noise.out_file
    outputs_['zoomed_report'] = mosaic_zoom.out_file
    outputs_['carpet_report'] = bigplot.out_file
    # fmt: on
    if True:  # wf_fft_spikes_detector: - disabled so output is always created
        mosaic_spikes = workflow.add(
            PlotSpikes(
                cmap="viridis",
                out_file="plot_spikes.svg",
                title="High-Frequency spikes",
            ),
            name="mosaic_spikes",
        )
        pass
        # fmt: off
        pass
        mosaic_spikes.inputs.in_file = in_ras
        mosaic_spikes.inputs.in_spikes = in_spikes
        mosaic_spikes.inputs.in_fft = in_fft
        outputs_['spikes_report'] = mosaic_spikes.out_file
        # fmt: on
    if not verbose:
        return workflow
    # Verbose-reporting goes here
    from pydra.tasks.nireports.interfaces import PlotContours
    from pydra.tasks.niworkflows.utils.connections import pop_file as _pop

    # fmt: off

    # fmt: on

    return tuple(outputs_)


def _carpet_parcellation(segmentation, crown_mask):
    """Generate the union of two masks."""
    from pathlib import Path
    import nibabel as nb
    import numpy as np

    img = nb.load(segmentation)
    lut = np.zeros((256,), dtype="uint8")
    lut[100:201] = 1  # Ctx GM
    lut[30:99] = 2  # dGM
    lut[1:11] = 3  # WM+CSF
    lut[255] = 4  # Cerebellum
    # Apply lookup table
    seg = lut[np.asanyarray(img.dataobj, dtype="uint16")]
    seg[np.asanyarray(nb.load(crown_mask).dataobj, dtype=int) > 0] = 5
    outimg = img.__class__(seg.astype("uint8"), img.affine, img.header)
    outimg.set_data_dtype("uint8")
    out_file = Path("segments.nii.gz").absolute()
    outimg.to_filename(out_file)
    return str(out_file)


def _get_tr(meta_dict):

    if isinstance(meta_dict, (list, tuple)):
        meta_dict = meta_dict[0]
    return meta_dict.get("RepetitionTime", None)


def spikes_mask(in_file, in_mask=None, out_file=None):
    """Calculate a mask in which check for :abbr:`EM (electromagnetic)` spikes."""
    import os.path as op
    import nibabel as nb
    import numpy as np
    from nilearn.image import mean_img
    from nilearn.plotting import plot_roi
    from scipy import ndimage as nd

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath(f"{fname}_spmask{ext}")
        out_plot = op.abspath(f"{fname}_spmask.pdf")
    in_4d_nii = nb.load(in_file)
    orientation = nb.aff2axcodes(in_4d_nii.affine)
    if in_mask:
        mask_data = np.asanyarray(nb.load(in_mask).dataobj)
        a = np.where(mask_data != 0)
        bbox = (
            np.max(a[0]) - np.min(a[0]),
            np.max(a[1]) - np.min(a[1]),
            np.max(a[2]) - np.min(a[2]),
        )
        longest_axis = np.argmax(bbox)
        # Input here is a binarized and intersected mask data from previous section
        dil_mask = nd.binary_dilation(
            mask_data, iterations=int(mask_data.shape[longest_axis] / 9)
        )
        rep = list(mask_data.shape)
        rep[longest_axis] = -1
        new_mask_2d = dil_mask.max(axis=longest_axis).reshape(rep)
        rep = [1, 1, 1]
        rep[longest_axis] = mask_data.shape[longest_axis]
        new_mask_3d = np.logical_not(np.tile(new_mask_2d, rep))
    else:
        new_mask_3d = np.zeros(in_4d_nii.shape[:3]) == 1
    if orientation[0] in ("L", "R"):
        new_mask_3d[0:2, :, :] = True
        new_mask_3d[-3:-1, :, :] = True
    else:
        new_mask_3d[:, 0:2, :] = True
        new_mask_3d[:, -3:-1, :] = True
    mask_nii = nb.Nifti1Image(
        new_mask_3d.astype(np.uint8), in_4d_nii.affine, in_4d_nii.header
    )
    mask_nii.to_filename(out_file)
    plot_roi(mask_nii, mean_img(in_4d_nii), output_file=out_plot)
    return out_file, out_plot
