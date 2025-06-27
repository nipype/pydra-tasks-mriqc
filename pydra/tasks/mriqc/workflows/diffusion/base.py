import attrs
import logging
from pathlib import Path
from pydra.compose import python, workflow
from pydra.tasks.mriqc.workflows.diffusion.output import init_dwi_report_wf
import typing as ty


logger = logging.getLogger(__name__)


@workflow.define(
    outputs=[
        "iqms_wf_out_file",
        "iqms_wf_noise_floor",
        "dwi_report_wf_snr_report",
        "dwi_report_wf_noise_report",
        "dwi_report_wf_fa_report",
        "dwi_report_wf_md_report",
        "dwi_report_wf_heatmap_report",
        "dwi_report_wf_spikes_report",
        "dwi_report_wf_carpet_report",
        "dwi_report_wf_bmask_report",
    ]
)
def dmri_qc_workflow(
    bvals: ty.Any = attrs.NOTHING,
    bvecs: ty.Any = attrs.NOTHING,
    exec_ants_float=False,
    exec_debug=False,
    exec_float32=True,
    exec_verbose_reports=False,
    exec_work_dir=None,
    in_file: ty.Any = attrs.NOTHING,
    name="dwiMRIQC",
    nipype_nprocs=12,
    nipype_omp_nthreads=12,
    qspace_neighbors: ty.Any = attrs.NOTHING,
    wf_biggest_file_gb=1,
    wf_fd_radius=50,
    wf_fd_thres=0.2,
    wf_inputs=None,
    wf_inputs_entities={},
    wf_inputs_metadata=None,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
) -> [
    "ty.Any",
    "ty.Any",
    "ty.Any",
    "ty.Any",
    "ty.Any",
    "ty.Any",
    "ty.Any",
    "ty.Any",
    "ty.Any",
    "ty.Any",
]:
    """
    Initialize the dMRI-QC workflow.

    .. workflow::

        import os.path as op
        from mriqc.workflows.diffusion.base import dmri_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = dmri_qc_workflow()

    """
    from pydra.tasks.afni.auto import Volreg

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    outputs_ = {
        "iqms_wf_out_file": attrs.NOTHING,
        "iqms_wf_noise_floor": attrs.NOTHING,
        "dwi_report_wf_snr_report": attrs.NOTHING,
        "dwi_report_wf_noise_report": attrs.NOTHING,
        "dwi_report_wf_fa_report": attrs.NOTHING,
        "dwi_report_wf_md_report": attrs.NOTHING,
        "dwi_report_wf_heatmap_report": attrs.NOTHING,
        "dwi_report_wf_spikes_report": attrs.NOTHING,
        "dwi_report_wf_carpet_report": attrs.NOTHING,
        "dwi_report_wf_bmask_report": attrs.NOTHING,
    }

    from pydra.tasks.mrtrix3.v3_0 import DwiDenoise
    from pydra.tasks.niworkflows.interfaces.header import SanitizeImage
    from pydra.tasks.niworkflows.interfaces.images import RobustAverage
    from pydra.tasks.mriqc.interfaces.diffusion import (
        CCSegmentation,
        CorrectSignalDrift,
        DiffusionModel,
        ExtractOrientations,
        NumberOfShells,
        PIESNO,
        ReadDWIMetadata,
        SpikingVoxelsMask,
        WeightedStat,
    )

    from pydra.tasks.mriqc.workflows.shared import synthstrip_wf as dmri_bmsk_workflow

    # Enable if necessary
    # mem_gb = wf_biggest_file_gb['dwi']
    dataset = wf_inputs["dwi"]
    metadata = wf_inputs_metadata["dwi"]
    entities = wf_inputs_entities["dwi"]
    message = "Building {modality} MRIQC workflow {detail}.".format(
        modality="diffusion",
        detail=f"for {len(dataset)} NIfTI files.",
    )
    logger.info(message)
    # Define workflow, inputs and outputs
    # 0. Get data, put it in RAS orientation

    sanitize = workflow.add(
        SanitizeImage(max_32bit=exec_float32, n_volumes_to_discard=0, in_file=in_file),
        name="sanitize",
    )
    # Workflow --------------------------------------------------------
    # Read metadata & bvec/bval, estimate number of shells, extract and split B0s

    shells = workflow.add(NumberOfShells(in_bvals=bvals), name="shells")
    get_lowb = workflow.add(
        ExtractOrientations(in_file=sanitize.out_file), name="get_lowb"
    )
    # Generate B0 reference
    dwi_ref = workflow.add(
        RobustAverage(mc_method=None, in_file=sanitize.out_file), name="dwi_ref"
    )
    hmc_b0 = workflow.add(
        Volreg(
            args="-Fourier -twopass",
            outputtype="NIFTI_GZ",
            zpad=4,
            basefile=dwi_ref.out_file,
            in_file=get_lowb.out_file,
        ),
        name="hmc_b0",
    )
    # Calculate brainmask
    dmri_bmsk = workflow.add(
        dmri_bmsk_workflow(
            omp_nthreads=nipype_omp_nthreads,
            in_files=dwi_ref.out_file,
            name="dmri_bmsk",
        )
    )
    # HMC: head motion correct
    hmcwf = workflow.add(
        hmc_workflow(wf_fd_radius=wf_fd_radius, in_bvec=bvecs, name="hmcwf")
    )
    get_hmc_shells = workflow.add(
        ExtractOrientations(
            in_bvec_file=bvecs, in_file=hmcwf.out_file, indices=shells.b_indices
        ),
        name="get_hmc_shells",
    )
    # Split shells and compute some stats
    averages = workflow.add(WeightedStat(in_weights=shells.b_masks), name="averages")
    stddev = workflow.add(
        WeightedStat(stat="std", in_weights=shells.b_masks), name="stddev"
    )
    dwidenoise = workflow.add(
        DwiDenoise(
            noise="noisemap.nii.gz",
            nthreads=nipype_omp_nthreads,
            mask=dmri_bmsk.out_mask,
        ),
        name="dwidenoise",
    )
    drift = workflow.add(
        CorrectSignalDrift(
            brainmask_file=dmri_bmsk.out_mask,
            bval_file=bvals,
            full_epi=sanitize.out_file,
            in_file=hmc_b0.out_file,
        ),
        name="drift",
    )
    sp_mask = workflow.add(
        SpikingVoxelsMask(
            b_masks=shells.b_masks,
            brain_mask=dmri_bmsk.out_mask,
            in_file=sanitize.out_file,
        ),
        name="sp_mask",
    )
    # Fit DTI/DKI model
    dwimodel = workflow.add(
        DiffusionModel(
            brain_mask=dmri_bmsk.out_mask,
            bvals=shells.out_data,
            bvec_file=bvecs,
            in_file=dwidenoise.out_file,
            n_shells=shells.n_shells,
        ),
        name="dwimodel",
    )
    # Calculate CC mask
    cc_mask = workflow.add(
        CCSegmentation(in_cfa=dwimodel.out_cfa, in_fa=dwimodel.out_fa), name="cc_mask"
    )
    # Run PIESNO noise estimation
    piesno = workflow.add(PIESNO(in_file=sanitize.out_file), name="piesno")
    # EPI to MNI registration
    spatial_norm = workflow.add(
        epi_mni_align(
            nipype_omp_nthreads=nipype_omp_nthreads,
            wf_species=wf_species,
            wf_template_id=wf_template_id,
            nipype_nprocs=nipype_nprocs,
            exec_debug=exec_debug,
            exec_ants_float=exec_ants_float,
            epi_mean=dwi_ref.out_file,
            epi_mask=dmri_bmsk.out_mask,
            name="spatial_norm",
        )
    )
    # Compute IQMs
    iqms_wf = workflow.add(
        compute_iqms(
            in_file=in_file,
            b_values_file=bvals,
            qspace_neighbors=qspace_neighbors,
            spikes_mask=sp_mask.out_mask,
            piesno_sigma=piesno.sigma,
            framewise_displacement=hmcwf.out_fd,
            in_bvec_rotated=hmcwf.out_bvec,
            in_bvec_diff=hmcwf.out_bvec_diff,
            in_fa=dwimodel.out_fa,
            in_cfa=dwimodel.out_cfa,
            in_fa_nans=dwimodel.out_fa_nans,
            in_fa_degenerate=dwimodel.out_fa_degenerate,
            in_md=dwimodel.out_md,
            brain_mask=dmri_bmsk.out_mask,
            cc_mask=cc_mask.out_mask,
            wm_mask=cc_mask.wm_finalmask,
            b_values_shells=shells.b_values,
            in_shells=get_hmc_shells.out_file,
            in_bvec=get_hmc_shells.out_bvec,
            in_noise=dwidenoise.noise,
            name="iqms_wf",
        )
    )
    # Generate outputs
    dwi_report_wf = workflow.add(
        init_dwi_report_wf(
            wf_species=wf_species,
            wf_biggest_file_gb=wf_biggest_file_gb,
            exec_verbose_reports=exec_verbose_reports,
            wf_fd_thres=wf_fd_thres,
            exec_work_dir=exec_work_dir,
            in_bdict=shells.b_dict,
            brain_mask=dmri_bmsk.out_mask,
            in_avgmap=averages.out_file,
            in_stdmap=stddev.out_file,
            in_epi=drift.out_full_file,
            in_fa=dwimodel.out_fa,
            in_md=dwimodel.out_md,
            in_parcellation=spatial_norm.epi_parc,
            name="dwi_report_wf",
        )
    )
    # fmt: off

    @python.define
    def shells_b_masks_to_dwi_ref_t_mask_callable(in_: ty.Any) -> ty.Any:
        return _first(in_)

    shells_b_masks_to_dwi_ref_t_mask_callable = workflow.add(shells_b_masks_to_dwi_ref_t_mask_callable(in_=shells.b_masks), name="shells_b_masks_to_dwi_ref_t_mask_callable")

    dwi_ref.inputs.t_mask = shells_b_masks_to_dwi_ref_t_mask_callable.out

    @python.define
    def shells_b_indices_to_get_lowb_indices_callable(in_: ty.Any) -> ty.Any:
        return _first(in_)

    shells_b_indices_to_get_lowb_indices_callable = workflow.add(shells_b_indices_to_get_lowb_indices_callable(in_=shells.b_indices), name="shells_b_indices_to_get_lowb_indices_callable")

    get_lowb.inputs.indices = shells_b_indices_to_get_lowb_indices_callable.out

    @python.define
    def shells_b_indices_to_drift_b0_ixs_callable(in_: ty.Any) -> ty.Any:
        return _first(in_)

    shells_b_indices_to_drift_b0_ixs_callable = workflow.add(shells_b_indices_to_drift_b0_ixs_callable(in_=shells.b_indices), name="shells_b_indices_to_drift_b0_ixs_callable")

    drift.inputs.b0_ixs = shells_b_indices_to_drift_b0_ixs_callable.out
    hmcwf.inputs.in_file = drift.out_full_file
    averages.inputs.in_file = drift.out_full_file
    stddev.inputs.in_file = drift.out_full_file

    @python.define
    def averages_out_file_to_hmcwf_reference_callable(in_: ty.Any) -> ty.Any:
        return _first(in_)

    averages_out_file_to_hmcwf_reference_callable = workflow.add(averages_out_file_to_hmcwf_reference_callable(in_=averages.out_file), name="averages_out_file_to_hmcwf_reference_callable")

    hmcwf.inputs.reference = averages_out_file_to_hmcwf_reference_callable.out
    dwidenoise.inputs.in_file = drift.out_full_file

    @python.define
    def averages_out_file_to_iqms_wf_in_b0_callable(in_: ty.Any) -> ty.Any:
        return _first(in_)

    averages_out_file_to_iqms_wf_in_b0_callable = workflow.add(averages_out_file_to_iqms_wf_in_b0_callable(in_=averages.out_file), name="averages_out_file_to_iqms_wf_in_b0_callable")

    iqms_wf.inputs.in_b0 = averages_out_file_to_iqms_wf_in_b0_callable.out
    # fmt: on
    outputs_["iqms_wf_out_file"] = iqms_wf.out_file
    outputs_["iqms_wf_noise_floor"] = iqms_wf.noise_floor
    outputs_["dwi_report_wf_noise_report"] = dwi_report_wf.noise_report
    outputs_["dwi_report_wf_md_report"] = dwi_report_wf.md_report
    outputs_["dwi_report_wf_bmask_report"] = dwi_report_wf.bmask_report
    outputs_["dwi_report_wf_snr_report"] = dwi_report_wf.snr_report
    outputs_["dwi_report_wf_carpet_report"] = dwi_report_wf.carpet_report
    outputs_["dwi_report_wf_fa_report"] = dwi_report_wf.fa_report
    outputs_["dwi_report_wf_spikes_report"] = dwi_report_wf.spikes_report
    outputs_["dwi_report_wf_heatmap_report"] = dwi_report_wf.heatmap_report

    return tuple(outputs_)


@workflow.define(outputs=["out_file", "out_fd", "out_bvec", "out_bvec_diff"])
def hmc_workflow(
    in_bvec: ty.Any = attrs.NOTHING,
    in_file: ty.Any = attrs.NOTHING,
    name="dMRI_HMC",
    reference: ty.Any = attrs.NOTHING,
    wf_fd_radius=50,
) -> ["ty.Any", "ty.Any", "ty.Any", "ty.Any"]:
    """
    Create a :abbr:`HMC (head motion correction)` workflow for dMRI.

    .. workflow::

        from mriqc.workflows.diffusion.base import hmc
        from mriqc.testing import mock_config
        with mock_config():
            wf = hmc()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import (
        FramewiseDisplacement,
    )

    outputs_ = {
        "out_file": attrs.NOTHING,
        "out_fd": attrs.NOTHING,
        "out_bvec": attrs.NOTHING,
        "out_bvec_diff": attrs.NOTHING,
    }

    from pydra.tasks.afni.auto import Volreg
    from pydra.tasks.mriqc.interfaces.diffusion import RotateVectors

    # calculate hmc parameters
    hmc = workflow.add(
        Volreg(
            args="-Fourier -twopass",
            outputtype="NIFTI_GZ",
            zpad=4,
            basefile=reference,
            in_file=in_file,
        ),
        name="hmc",
    )
    bvec_rot = workflow.add(
        RotateVectors(
            in_file=in_bvec, reference=reference, transforms=hmc.oned_matrix_save
        ),
        name="bvec_rot",
    )
    # Compute the frame-wise displacement
    fdnode = workflow.add(
        FramewiseDisplacement(
            normalize=False,
            parameter_source="AFNI",
            radius=wf_fd_radius,
            in_file=hmc.oned_file,
        ),
        name="fdnode",
    )
    # fmt: off
    outputs_['out_file'] = hmc.out_file
    outputs_['out_fd'] = fdnode.out_file
    outputs_['out_bvec'] = bvec_rot.out_bvec
    outputs_['out_bvec_diff'] = bvec_rot.out_diff
    # fmt: on

    return tuple(outputs_)


@workflow.define(outputs=["epi_parc", "epi_mni", "report"])
def epi_mni_align(
    epi_mask: ty.Any = attrs.NOTHING,
    epi_mean: ty.Any = attrs.NOTHING,
    exec_ants_float=False,
    exec_debug=False,
    name="SpatialNormalization",
    nipype_nprocs=12,
    nipype_omp_nthreads=12,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
) -> ["ty.Any", "ty.Any", "ty.Any"]:
    """
    Estimate the transform that maps the EPI space into MNI152NLin2009cAsym.

    The input epi_mean is the averaged and brain-masked EPI timeseries

    Returns the EPI mean resampled in MNI space (for checking out registration) and
    the associated "lobe" parcellation in EPI space.

    .. workflow::

        from mriqc.workflows.diffusion.base import epi_mni_align
        from mriqc.testing import mock_config
        with mock_config():
            wf = epi_mni_align()

    """
    from pydra.tasks.ants.auto import ApplyTransforms, N4BiasFieldCorrection

    outputs_ = {
        "epi_parc": attrs.NOTHING,
        "epi_mni": attrs.NOTHING,
        "report": attrs.NOTHING,
    }

    from pydra.tasks.niworkflows.interfaces.reportlets.registration import (
        SpatialNormalizationRPT as RobustMNINormalization,
    )
    from templateflow.api import get as get_template

    # Get settings
    testing = exec_debug
    n_procs = nipype_nprocs
    ants_nthreads = nipype_omp_nthreads

    n4itk = workflow.add(
        N4BiasFieldCorrection(copy_header=True, dimension=3, input_image=epi_mean),
        name="n4itk",
    )
    norm = workflow.add(
        RobustMNINormalization(
            explicit_masking=False,
            flavor="testing" if testing else "precise",
            float=exec_ants_float,
            generate_report=True,
            moving="boldref",
            num_threads=ants_nthreads,
            reference="boldref",
            template=wf_template_id,
            moving_image=n4itk.output_image,
        ),
        name="norm",
    )
    if wf_species.lower() == "human":
        norm.inputs.inputs.reference_image = str(
            get_template(wf_template_id, resolution=2, suffix="boldref")
        )
        norm.inputs.inputs.reference_mask = str(
            get_template(
                wf_template_id,
                resolution=2,
                desc="brain",
                suffix="mask",
            )
        )
    # adapt some population-specific settings
    else:
        from nirodents.workflows.brainextraction import _bspline_grid

        n4itk.inputs.inputs.shrink_factor = 1
        n4itk.inputs.inputs.n_iterations = [50] * 4
        norm.inputs.inputs.reference_image = str(
            get_template(wf_template_id, suffix="T2w")
        )
        norm.inputs.inputs.reference_mask = str(
            get_template(
                wf_template_id,
                desc="brain",
                suffix="mask",
            )[0]
        )
        bspline_grid = workflow.add(
            FunctionTask(func=_bspline_grid), name="bspline_grid"
        )
        # fmt: off
        bspline_grid.inputs.in_file = epi_mean
        n4itk.inputs.args = bspline_grid.out
        # fmt: on
    # Warp segmentation into EPI space
    invt = workflow.add(
        ApplyTransforms(
            default_value=0,
            dimension=3,
            float=True,
            interpolation="MultiLabel",
            reference_image=epi_mean,
            transforms=norm.inverse_composite_transform,
        ),
        name="invt",
    )
    if wf_species.lower() == "human":
        invt.inputs.inputs.input_image = str(
            get_template(
                wf_template_id,
                resolution=1,
                desc="carpet",
                suffix="dseg",
            )
        )
    else:
        invt.inputs.inputs.input_image = str(
            get_template(
                wf_template_id,
                suffix="dseg",
            )[-1]
        )
    # fmt: off
    outputs_['epi_parc'] = invt.output_image
    outputs_['epi_mni'] = norm.warped_image
    outputs_['report'] = norm.out_report
    # fmt: on
    if wf_species.lower() == "human":
        norm.inputs.moving_mask = epi_mask

    return tuple(outputs_)


@workflow.define(outputs=["out_file", "noise_floor"])
def compute_iqms(
    b_values_file: ty.Any = attrs.NOTHING,
    b_values_shells: ty.Any = attrs.NOTHING,
    brain_mask: ty.Any = attrs.NOTHING,
    cc_mask: ty.Any = attrs.NOTHING,
    framewise_displacement: ty.Any = attrs.NOTHING,
    in_b0: ty.Any = attrs.NOTHING,
    in_bvec: ty.Any = attrs.NOTHING,
    in_bvec_diff: ty.Any = attrs.NOTHING,
    in_bvec_rotated: ty.Any = attrs.NOTHING,
    in_cfa: ty.Any = attrs.NOTHING,
    in_fa: ty.Any = attrs.NOTHING,
    in_fa_degenerate: ty.Any = attrs.NOTHING,
    in_fa_nans: ty.Any = attrs.NOTHING,
    in_file: ty.Any = attrs.NOTHING,
    in_md: ty.Any = attrs.NOTHING,
    in_noise: ty.Any = attrs.NOTHING,
    in_shells: ty.Any = attrs.NOTHING,
    name="ComputeIQMs",
    piesno_sigma: ty.Any = attrs.NOTHING,
    qspace_neighbors: ty.Any = attrs.NOTHING,
    spikes_mask: ty.Any = attrs.NOTHING,
    wm_mask: ty.Any = attrs.NOTHING,
) -> ["ty.Any", "ty.Any"]:
    """
    Initialize the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.diffusion.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from pydra.tasks.mriqc.interfaces import IQMFileSink

    outputs_ = {
        "out_file": attrs.NOTHING,
        "noise_floor": attrs.NOTHING,
    }

    from pydra.tasks.mriqc.interfaces.diffusion import DiffusionQC
    from pydra.tasks.mriqc.interfaces.reports import AddProvenance

    # from mriqc.workflows.utils import _tofloat, get_fwhmx

    estimate_sigma = workflow.add(
        FunctionTask(func=_estimate_sigma, in_file=in_noise, mask=brain_mask),
        name="estimate_sigma",
    )
    measures = workflow.add(
        DiffusionQC(
            brain_mask=brain_mask,
            cc_mask=cc_mask,
            in_b0=in_b0,
            in_bval_file=b_values_file,
            in_bvec=in_bvec,
            in_bvec_diff=in_bvec_diff,
            in_bvec_rotated=in_bvec_rotated,
            in_cfa=in_cfa,
            in_fa=in_fa,
            in_fa_degenerate=in_fa_degenerate,
            in_fa_nans=in_fa_nans,
            in_fd=framewise_displacement,
            in_file=in_file,
            in_md=in_md,
            in_shells=in_shells,
            in_shells_bval=b_values_shells,
            piesno_sigma=piesno_sigma,
            qspace_neighbors=qspace_neighbors,
            spikes_mask=spikes_mask,
            wm_mask=wm_mask,
        ),
        name="measures",
    )

    # Save to JSON file

    # fmt: off



    outputs_['out_file'] = measures.out_qc
    outputs_['noise_floor'] = estimate_sigma.out
    # fmt: on

    return tuple(outputs_)


def _bvals_report(in_file):

    import numpy as np

    bvals = [
        round(float(val), 2) for val in np.unique(np.round(np.loadtxt(in_file), 2))
    ]
    if len(bvals) > 10:
        return "Likely DSI"
    return bvals


def _estimate_sigma(in_file, mask):

    import nibabel as nb
    from numpy import median

    msk = nb.load(mask).get_fdata() > 0.5
    return round(
        float(median(nb.load(in_file).get_fdata()[msk])),
        6,
    )


def _filter_metadata(
    in_dict,
    keys=(
        "global",
        "dcmmeta_affine",
        "dcmmeta_reorient_transform",
        "dcmmeta_shape",
        "dcmmeta_slice_dim",
        "dcmmeta_version",
        "time",
    ),
):
    """Drop large and partially redundant objects generated by dcm2niix."""
    for key in keys:
        in_dict.pop(key, None)
    return in_dict


def _first(inlist):

    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist
