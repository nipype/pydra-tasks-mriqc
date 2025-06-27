import attrs
from fileformats.field import Integer
import logging
from pydra.tasks.mriqc.workflows.functional.output import init_func_report_wf
from pydra.tasks.niworkflows.utils.connections import pop_file as _pop
from pathlib import Path
from pydra.compose import python, workflow
from pydra.tasks.niworkflows.utils.connections import pop_file as _pop
import typing as ty


logger = logging.getLogger(__name__)


@workflow.define(outputs=["out_file"])
def fmri_bmsk_workflow(in_file: ty.Any = attrs.NOTHING, name="fMRIBrainMask") -> ty.Any:
    """
    Compute a brain mask for the input :abbr:`fMRI (functional MRI)` dataset.

    .. workflow::

        from mriqc.workflows.functional.base import fmri_bmsk_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = fmri_bmsk_workflow()


    """
    from pydra.tasks.afni.auto import Automask

    outputs_ = {
        "out_file": attrs.NOTHING,
    }

    afni_msk = workflow.add(
        Automask(outputtype="NIFTI_GZ", in_file=in_file), name="afni_msk"
    )
    # Connect brain mask extraction
    # fmt: off
    outputs_['out_file'] = afni_msk.out_file
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
) -> tuple[ty.Any, ty.Any, ty.Any]:
    """
    Estimate the transform that maps the EPI space into MNI152NLin2009cAsym.

    The input epi_mean is the averaged and brain-masked EPI timeseries

    Returns the EPI mean resampled in MNI space (for checking out registration) and
    the associated "lobe" parcellation in EPI space.

    .. workflow::

        from mriqc.workflows.functional.base import epi_mni_align
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
        bspline_grid = workflow.add(python.define(_bspline_grid)(), name="bspline_grid")
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


@workflow.define(outputs=["out_file", "mpars", "out_fd"])
def hmc(
    fd_radius: ty.Any = attrs.NOTHING,
    in_file: ty.Any = attrs.NOTHING,
    name="fMRI_HMC",
    omp_nthreads=None,
    wf_biggest_file_gb=1,
    wf_deoblique=False,
    wf_despike=False,
) -> tuple[ty.Any, ty.Any, ty.Any]:
    """
    Create a :abbr:`HMC (head motion correction)` workflow for fMRI.

    .. workflow::

        from mriqc.workflows.functional.base import hmc
        from mriqc.testing import mock_config
        with mock_config():
            wf = hmc()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import (
        FramewiseDisplacement,
    )

    outputs_ = {
        "out_file": attrs.NOTHING,
        "mpars": attrs.NOTHING,
        "out_fd": attrs.NOTHING,
    }

    from pydra.tasks.afni.auto import Despike, Refit, Volreg

    mem_gb = wf_biggest_file_gb["bold"]

    # calculate hmc parameters
    estimate_hm = workflow.add(
        Volreg(args="-Fourier -twopass", outputtype="NIFTI_GZ", zpad=4),
        name="estimate_hm",
    )
    # Compute the frame-wise displacement
    fdnode = workflow.add(
        FramewiseDisplacement(
            normalize=False,
            parameter_source="AFNI",
            in_file=estimate_hm.oned_file,
            radius=fd_radius,
        ),
        name="fdnode",
    )
    # Apply transforms to other echos
    apply_hmc = workflow.add(
        python.define(
            _apply_transforms,
            inputs={"in_file": ty.Any, "in_xfm": ty.Any, "max_concurrent": ty.Any},
        )(in_xfm=estimate_hm.oned_matrix_save),
        name="apply_hmc",
    )
    apply_hmc.inputs.inputs.max_concurrent = 4
    # fmt: off
    outputs_['out_file'] = apply_hmc.out
    outputs_['mpars'] = estimate_hm.oned_file
    outputs_['out_fd'] = fdnode.out_file
    # fmt: on
    if not (wf_despike or wf_deoblique):
        # fmt: off
        estimate_hm.inputs.in_file = in_file
        apply_hmc.inputs.in_file = in_file
        # fmt: on
        return workflow
    # despiking, and deoblique
    deoblique_node = workflow.add(Refit(deoblique=True), name="deoblique_node")
    despike_node = workflow.add(Despike(outputtype="NIFTI_GZ"), name="despike_node")
    if wf_despike and wf_deoblique:
        # fmt: off
        despike_node.inputs.in_file = in_file
        deoblique_node.inputs.in_file = despike_node.out_file

        @python.define
        def deoblique_node_out_file_to_estimate_hm_in_file_callable(in_: ty.Any) -> ty.Any:
            return _pop(in_)

        deoblique_node_out_file_to_estimate_hm_in_file_callable = workflow.add(deoblique_node_out_file_to_estimate_hm_in_file_callable(in_=deoblique_node.out_file), name="deoblique_node_out_file_to_estimate_hm_in_file_callable")

        estimate_hm.inputs.in_file = deoblique_node_out_file_to_estimate_hm_in_file_callable.out
        apply_hmc.inputs.in_file = deoblique_node.out_file
        # fmt: on
    elif wf_despike:
        # fmt: off
        despike_node.inputs.in_file = in_file

        @python.define
        def despike_node_out_file_to_estimate_hm_in_file_callable(in_: ty.Any) -> ty.Any:
            return _pop(in_)

        despike_node_out_file_to_estimate_hm_in_file_callable = workflow.add(despike_node_out_file_to_estimate_hm_in_file_callable(in_=despike_node.out_file), name="despike_node_out_file_to_estimate_hm_in_file_callable")

        estimate_hm.inputs.in_file = despike_node_out_file_to_estimate_hm_in_file_callable.out
        apply_hmc.inputs.in_file = despike_node.out_file
        # fmt: on
    elif wf_deoblique:
        # fmt: off
        deoblique_node.inputs.in_file = in_file

        @python.define
        def deoblique_node_out_file_to_estimate_hm_in_file_callable(in_: ty.Any) -> ty.Any:
            return _pop(in_)

        deoblique_node_out_file_to_estimate_hm_in_file_callable = workflow.add(deoblique_node_out_file_to_estimate_hm_in_file_callable(in_=deoblique_node.out_file), name="deoblique_node_out_file_to_estimate_hm_in_file_callable")

        estimate_hm.inputs.in_file = deoblique_node_out_file_to_estimate_hm_in_file_callable.out
        apply_hmc.inputs.in_file = deoblique_node.out_file
        # fmt: on
    else:
        raise NotImplementedError

    return tuple(outputs_)


def _apply_transforms(in_file, in_xfm, max_concurrent):

    from pathlib import Path
    from nitransforms.linear import load
    from nitransforms.resampling import apply
    from pydra.tasks.mriqc.utils.bids import derive_bids_fname

    realigned = apply(
        load(in_xfm, fmt="afni", reference=in_file, moving=in_file),
        in_file,
        dtype_width=4,
        serialize_nvols=2,
        max_concurrent=max_concurrent,
        mode="reflect",
    )
    out_file = derive_bids_fname(
        in_file,
        entity="desc-realigned",
        newpath=Path.cwd(),
        absolute=True,
    )
    realigned.to_filename(out_file)
    return str(out_file)


@workflow.define(
    outputs=["out_file", "spikes", "fft", "spikes_num", "outliers", "dvars"]
)
def compute_iqms(
    brainmask: ty.Any = attrs.NOTHING,
    epi_mean: ty.Any = attrs.NOTHING,
    fd_thres: ty.Any = attrs.NOTHING,
    hmc_epi: ty.Any = attrs.NOTHING,
    hmc_fd: ty.Any = attrs.NOTHING,
    in_ras: ty.Any = attrs.NOTHING,
    in_tsnr: ty.Any = attrs.NOTHING,
    name="ComputeIQMs",
    wf_biggest_file_gb=1,
    wf_fft_spikes_detector=False,
) -> tuple[ty.Any, ty.Any, ty.Any, Integer, ty.Any, ty.Any]:
    """
    Initialize the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.functional.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import ComputeDVARS

    outputs_ = {
        "out_file": attrs.NOTHING,
        "spikes": attrs.NOTHING,
        "fft": attrs.NOTHING,
        "spikes_num": attrs.NOTHING,
        "outliers": attrs.NOTHING,
        "dvars": attrs.NOTHING,
    }

    from pydra.tasks.afni.auto import OutlierCount, QualityIndex
    from pydra.tasks.mriqc.interfaces import (
        DerivativesDataSink,
        FunctionalQC,
        GatherTimeseries,
        IQMFileSink,
    )
    from pydra.tasks.mriqc.interfaces.reports import AddProvenance
    from pydra.tasks.mriqc.interfaces.transitional import GCOR
    from pydra.tasks.mriqc.workflows.utils import _tofloat, get_fwhmx

    mem_gb = wf_biggest_file_gb["bold"]

    # Set FD threshold

    # Compute DVARS
    dvnode = workflow.add(
        ComputeDVARS(
            save_all=True, save_plot=False, in_file=hmc_epi, in_mask=brainmask
        ),
        name="dvnode",
    )
    # AFNI quality measures
    fwhm = workflow.add(fwhm_task, name="fwhm")
    fwhm.inputs.inputs.acf = True  # Only AFNI >= 16
    outliers = workflow.add(
        OutlierCount(
            fraction=True, out_file="outliers.out", in_file=hmc_epi, mask=brainmask
        ),
        name="outliers",
    )

    measures = workflow.add(
        FunctionalQC(
            fd_thres=fd_thres,
            in_epi=epi_mean,
            in_fd=hmc_fd,
            in_hmc=hmc_epi,
            in_mask=brainmask,
            in_tsnr=in_tsnr,
        ),
        name="measures",
    )

    # fmt: off
    outputs_['dvars'] = dvnode.out_all

    @python.define
    def fwhm_fwhm_to_measures_in_fwhm_callable(in_: ty.Any) -> ty.Any:
        return _tofloat(in_)

    fwhm_fwhm_to_measures_in_fwhm_callable = workflow.add(fwhm_fwhm_to_measures_in_fwhm_callable(in_=fwhm.fwhm), name="fwhm_fwhm_to_measures_in_fwhm_callable")

    measures.inputs.in_fwhm = fwhm_fwhm_to_measures_in_fwhm_callable.out
    outputs_['outliers'] = outliers.out_file
    # fmt: on

    # Save to JSON file

    # Save timeseries TSV file

    # fmt: off


    outputs_['out_file'] = measures.out_qc

    # fmt: on
    # FFT spikes finder
    if True:  # wf_fft_spikes_detector: - disabled to ensure all outputs are generated
        from pydra.tasks.mriqc.workflows.utils import slice_wise_fft

        spikes_fft = workflow.add(
            python.define(
                slice_wise_fft,
                inputs={"in_file": ty.Any},
                outputs={"n_spikes": ty.Any, "out_spikes": ty.Any, "out_fft": ty.Any},
            )(),
            name="spikes_fft",
        )
        # fmt: off
        spikes_fft.inputs.in_file = in_ras
        outputs_['spikes'] = spikes_fft.out_spikes
        outputs_['fft'] = spikes_fft.out_fft
        outputs_['spikes_num'] = spikes_fft.n_spikes
        # fmt: on

    return tuple(outputs_)


def _parse_tout(in_file):

    if isinstance(in_file, (list, tuple)):
        return (
            [_parse_tout(f) for f in in_file]
            if len(in_file) > 1
            else _parse_tout(in_file[0])
        )
    import numpy as np

    data = np.loadtxt(in_file)  # pylint: disable=no-member
    return data.mean()


def _parse_tqual(in_file):

    if isinstance(in_file, (list, tuple)):
        return (
            [_parse_tqual(f) for f in in_file]
            if len(in_file) > 1
            else _parse_tqual(in_file[0])
        )
    import numpy as np

    with open(in_file) as fin:
        lines = fin.readlines()
    return np.mean([float(line.strip()) for line in lines if not line.startswith("++")])


@workflow.define(
    outputs=[
        "ema_report",
        "iqmswf_out_file",
        "iqmswf_spikes",
        "iqmswf_fft",
        "iqmswf_spikes_num",
        "iqmswf_outliers",
        "iqmswf_dvars",
        "func_report_wf_mean_report",
        "func_report_wf_stdev_report",
        "func_report_wf_background_report",
        "func_report_wf_zoomed_report",
        "func_report_wf_carpet_report",
        "func_report_wf_spikes_report",
    ]
)
def fmri_qc_workflow(
    exec_ants_float=False,
    exec_debug=False,
    exec_float32=True,
    exec_no_sub=False,
    exec_verbose_reports=False,
    exec_work_dir=None,
    in_file: ty.Any = attrs.NOTHING,
    metadata: ty.Any = attrs.NOTHING,
    name="funcMRIQC",
    nipype_nprocs=12,
    nipype_omp_nthreads=12,
    wf_biggest_file_gb=1,
    wf_deoblique=False,
    wf_despike=False,
    wf_fd_radius=50,
    wf_fft_spikes_detector=False,
    wf_inputs=None,
    wf_inputs_entities={},
    wf_inputs_metadata=None,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
) -> tuple[
    ty.Any,
    ty.Any,
    ty.Any,
    ty.Any,
    ty.Any,
    ty.Any,
    ty.Any,
    ty.Any,
    ty.Any,
    ty.Any,
    ty.Any,
    ty.Any,
    ty.Any,
]:
    """
    Initialize the (f)MRIQC workflow.

    .. workflow::

        import os.path as op
        from mriqc.workflows.functional.base import fmri_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = fmri_qc_workflow()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import (
        NonSteadyStateDetector,
        TSNR,
    )

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    outputs_ = {
        "ema_report": attrs.NOTHING,
        "iqmswf_out_file": attrs.NOTHING,
        "iqmswf_spikes": attrs.NOTHING,
        "iqmswf_fft": attrs.NOTHING,
        "iqmswf_spikes_num": attrs.NOTHING,
        "iqmswf_outliers": attrs.NOTHING,
        "iqmswf_dvars": attrs.NOTHING,
        "func_report_wf_mean_report": attrs.NOTHING,
        "func_report_wf_stdev_report": attrs.NOTHING,
        "func_report_wf_background_report": attrs.NOTHING,
        "func_report_wf_zoomed_report": attrs.NOTHING,
        "func_report_wf_carpet_report": attrs.NOTHING,
        "func_report_wf_spikes_report": attrs.NOTHING,
    }

    from pydra.tasks.afni.auto import TStat
    from pydra.tasks.niworkflows.interfaces.header import SanitizeImage
    from pydra.tasks.mriqc.interfaces.functional import SelectEcho

    mem_gb = wf_biggest_file_gb["bold"]
    dataset = wf_inputs["bold"]
    metadata = wf_inputs_metadata["bold"]
    entities = wf_inputs_entities["bold"]
    message = "Building {modality} MRIQC workflow {detail}.".format(
        modality="functional",
        detail=f"for {len(dataset)} BOLD runs.",
    )
    logger.info(message)
    # Define workflow, inputs and outputs
    # 0. Get data, put it in RAS orientation

    pick_echo = workflow.add(
        SelectEcho(in_files=in_file, metadata=metadata), name="pick_echo"
    )
    non_steady_state_detector = workflow.add(
        NonSteadyStateDetector(in_file=pick_echo.out_file),
        name="non_steady_state_detector",
    )
    sanitize = workflow.add(
        SanitizeImage(
            max_32bit=exec_float32,
            in_file=in_file,
            n_volumes_to_discard=non_steady_state_detector.n_volumes_to_discard,
        ),
        name="sanitize",
    )
    # Workflow --------------------------------------------------------
    # 1. HMC: head motion correct
    hmcwf = workflow.add(
        hmc(
            omp_nthreads=nipype_omp_nthreads,
            wf_deoblique=wf_deoblique,
            wf_biggest_file_gb=wf_biggest_file_gb,
            wf_despike=wf_despike,
            in_file=sanitize.out_file,
            name="hmcwf",
        )
    )
    # Set HMC settings
    hmcwf.inputs.inputs.inputnode.fd_radius = wf_fd_radius
    # 2. Compute mean fmri
    mean = workflow.add(
        TStat(options="-mean", outputtype="NIFTI_GZ", in_file=hmcwf.out_file),
        name="mean",
    )
    # Compute TSNR using nipype implementation
    tsnr = workflow.add(TSNR(in_file=hmcwf.out_file), name="tsnr")
    # EPI to MNI registration
    ema = workflow.add(
        epi_mni_align(
            wf_species=wf_species,
            nipype_omp_nthreads=nipype_omp_nthreads,
            nipype_nprocs=nipype_nprocs,
            wf_template_id=wf_template_id,
            exec_ants_float=exec_ants_float,
            exec_debug=exec_debug,
            name="ema",
        )
    )
    # 7. Compute IQMs
    iqmswf = workflow.add(
        compute_iqms(
            wf_biggest_file_gb=wf_biggest_file_gb,
            wf_fft_spikes_detector=wf_fft_spikes_detector,
            in_ras=sanitize.out_file,
            epi_mean=mean.out_file,
            hmc_epi=hmcwf.out_file,
            hmc_fd=hmcwf.out_fd,
            in_tsnr=tsnr.tsnr_file,
            name="iqmswf",
        )
    )
    # Reports
    func_report_wf = workflow.add(
        init_func_report_wf(
            exec_verbose_reports=exec_verbose_reports,
            exec_work_dir=exec_work_dir,
            wf_species=wf_species,
            wf_biggest_file_gb=wf_biggest_file_gb,
            wf_fft_spikes_detector=wf_fft_spikes_detector,
            meta_sidecar=metadata,
            in_ras=sanitize.out_file,
            epi_mean=mean.out_file,
            in_stddev=tsnr.stddev_file,
            hmc_fd=hmcwf.out_fd,
            hmc_epi=hmcwf.out_file,
            epi_parc=ema.epi_parc,
            name="func_report_wf",
        )
    )
    # fmt: off

    @python.define
    def mean_out_file_to_ema_epi_mean_callable(in_: ty.Any) -> ty.Any:
        return _pop(in_)

    mean_out_file_to_ema_epi_mean_callable = workflow.add(mean_out_file_to_ema_epi_mean_callable(in_=mean.out_file), name="mean_out_file_to_ema_epi_mean_callable")

    ema.inputs.epi_mean = mean_out_file_to_ema_epi_mean_callable.out

    # fmt: on
    if wf_fft_spikes_detector:
        # fmt: off
        outputs_['iqmswf_spikes'] = iqmswf.spikes
        outputs_['iqmswf_fft'] = iqmswf.fft
        # fmt: on
    # population specific changes to brain masking
    if wf_species == "human":
        from pydra.tasks.mriqc.workflows.shared import (
            synthstrip_wf as fmri_bmsk_workflow,
        )

        skullstrip_epi = workflow.add(
            fmri_bmsk_workflow(omp_nthreads=nipype_omp_nthreads, name="skullstrip_epi")
        )
        # fmt: off

        @python.define
        def mean_out_file_to_skullstrip_epi_in_files_callable(in_: ty.Any) -> ty.Any:
            return _pop(in_)

        mean_out_file_to_skullstrip_epi_in_files_callable = workflow.add(mean_out_file_to_skullstrip_epi_in_files_callable(in_=mean.out_file), name="mean_out_file_to_skullstrip_epi_in_files_callable")

        skullstrip_epi.inputs.in_files = mean_out_file_to_skullstrip_epi_in_files_callable.out
        ema.inputs.epi_mask = skullstrip_epi.out_mask
        iqmswf.inputs.brainmask = skullstrip_epi.out_mask
        func_report_wf.inputs.brainmask = skullstrip_epi.out_mask
        # fmt: on
    else:
        from pydra.tasks.mriqc.workflows.anatomical.base import _binarize

        binarise_labels = workflow.add(
            python.define(
                _binarize,
                inputs={"in_file": ty.Any, "threshold": ty.Any},
                outputs={"out_file": ty.Any},
            )(),
            name="binarise_labels",
        )
        # fmt: off
        binarise_labels.inputs.in_file = ema.epi_parc
        iqmswf.inputs.brainmask = binarise_labels.out_file
        func_report_wf.inputs.brainmask = binarise_labels.out_file
        # fmt: on
    # Upload metrics
    if not exec_no_sub:
        from pydra.tasks.mriqc.interfaces.webapi import UploadIQMs

        pass
        # fmt: off
        outputs_['iqmswf_out_file'] = iqmswf.out_file
        # fmt: on
    outputs_["ema_report"] = ema.report
    outputs_["iqmswf_outliers"] = iqmswf.outliers
    outputs_["iqmswf_spikes"] = iqmswf.spikes
    outputs_["iqmswf_out_file"] = iqmswf.out_file
    outputs_["iqmswf_spikes_num"] = iqmswf.spikes_num
    outputs_["iqmswf_fft"] = iqmswf.fft
    outputs_["iqmswf_dvars"] = iqmswf.dvars
    outputs_["func_report_wf_carpet_report"] = func_report_wf.carpet_report
    outputs_["func_report_wf_background_report"] = func_report_wf.background_report
    outputs_["func_report_wf_spikes_report"] = func_report_wf.spikes_report
    outputs_["func_report_wf_mean_report"] = func_report_wf.mean_report
    outputs_["func_report_wf_stdev_report"] = func_report_wf.stdev_report
    outputs_["func_report_wf_zoomed_report"] = func_report_wf.zoomed_report

    return tuple(outputs_)
