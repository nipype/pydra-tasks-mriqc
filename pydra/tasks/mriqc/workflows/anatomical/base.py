import attrs
from itertools import chain
import logging
from pathlib import Path
from pydra.compose import python, workflow
from pydra.tasks.mriqc.interfaces import (
    ArtifactMask,
    ComputeQI2,
    ConformImage,
    RotationMask,
    StructuralQC,
)
from pydra.tasks.mriqc.workflows.anatomical.output import init_anat_report_wf
from pydra.tasks.mriqc.workflows.utils import get_fwhmx
from pydra.tasks.niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
)
from templateflow.api import get as get_template
import typing as ty


logger = logging.getLogger(__name__)


@workflow.define(
    outputs=[
        "norm_report",
        "iqmswf_noise_report",
        "anat_report_wf_zoom_report",
        "anat_report_wf_bg_report",
        "anat_report_wf_segm_report",
        "anat_report_wf_bmask_report",
        "anat_report_wf_artmask_report",
        "anat_report_wf_airmask_report",
        "anat_report_wf_headmask_report",
    ]
)
def anat_qc_workflow(
    exec_ants_float=False,
    exec_debug=False,
    exec_no_sub=False,
    exec_verbose_reports=False,
    exec_work_dir=None,
    in_file: ty.Any = attrs.NOTHING,
    modality: ty.Any = attrs.NOTHING,
    name="anatMRIQC",
    nipype_omp_nthreads=12,
    wf_biggest_file_gb=1,
    wf_inputs=None,
    wf_inputs_entities={},
    wf_inputs_metadata=None,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
) -> tuple[ty.Any, ty.Any, ty.Any, ty.Any, ty.Any, ty.Any, ty.Any, ty.Any, ty.Any]:
    """
    One-subject-one-session-one-run pipeline to extract the NR-IQMs from
    anatomical images

    .. workflow::

        import os.path as op
        from mriqc.workflows.anatomical.base import anat_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = anat_qc_workflow()

    """
    from pydra.tasks.mriqc.workflows.shared import synthstrip_wf

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    outputs_ = {
        "norm_report": attrs.NOTHING,
        "iqmswf_noise_report": attrs.NOTHING,
        "anat_report_wf_zoom_report": attrs.NOTHING,
        "anat_report_wf_bg_report": attrs.NOTHING,
        "anat_report_wf_segm_report": attrs.NOTHING,
        "anat_report_wf_bmask_report": attrs.NOTHING,
        "anat_report_wf_artmask_report": attrs.NOTHING,
        "anat_report_wf_airmask_report": attrs.NOTHING,
        "anat_report_wf_headmask_report": attrs.NOTHING,
    }

    # Enable if necessary
    # mem_gb = max(
    # wf_biggest_file_gb['t1w'],
    # wf_biggest_file_gb['t2w'],
    # )
    dataset = list(
        chain(
            wf_inputs.get("t1w", []),
            wf_inputs.get("t2w", []),
        )
    )
    metadata = list(
        chain(
            wf_inputs_metadata.get("t1w", []),
            wf_inputs_metadata.get("t2w", []),
        )
    )
    entities = list(
        chain(
            wf_inputs_entities.get("t1w", []),
            wf_inputs_entities.get("t2w", []),
        )
    )
    message = "Building {modality} MRIQC workflow {detail}.".format(
        modality="anatomical",
        detail=f"for {len(dataset)} NIfTI files.",
    )
    logger.info(message)
    # Initialize workflow
    # Define workflow, inputs and outputs
    # 0. Get data

    # 1. Reorient anatomical image
    to_ras = workflow.add(
        ConformImage(check_dtype=False, in_file=in_file), name="to_ras"
    )
    # 2. species specific skull-stripping
    if wf_species.lower() == "human":
        skull_stripping = workflow.add(
            synthstrip_wf(
                omp_nthreads=nipype_omp_nthreads,
                in_files=to_ras.out_file,
                name="skull_stripping",
            )
        )
        ss_bias_field = "outputnode.bias_image"
    else:
        from nirodents.workflows.brainextraction import init_rodent_brain_extraction_wf

        skull_stripping = init_rodent_brain_extraction_wf(template_id=wf_template_id)
        ss_bias_field = "final_n4.bias_image"
    # 3. Head mask
    hmsk = workflow.add(
        headmsk_wf(omp_nthreads=nipype_omp_nthreads, wf_species=wf_species, name="hmsk")
    )
    # 4. Spatial Normalization, using ANTs
    norm = workflow.add(
        spatial_normalization(
            nipype_omp_nthreads=nipype_omp_nthreads,
            wf_template_id=wf_template_id,
            exec_ants_float=exec_ants_float,
            exec_debug=exec_debug,
            wf_species=wf_species,
            modality=modality,
            name="norm",
        )
    )
    # 5. Air mask (with and without artifacts)
    amw = workflow.add(
        airmsk_wf(
            ind2std_xfm=norm.ind2std_xfm,
            in_file=to_ras.out_file,
            head_mask=hmsk.out_file,
            name="amw",
        )
    )
    # 6. Brain tissue segmentation
    bts = workflow.add(
        init_brain_tissue_segmentation(
            nipype_omp_nthreads=nipype_omp_nthreads,
            std_tpms=norm.out_tpms,
            in_file=hmsk.out_denoised,
            name="bts",
        )
    )
    # 7. Compute IQMs
    iqmswf = workflow.add(
        compute_iqms(
            wf_species=wf_species,
            std_tpms=norm.out_tpms,
            in_ras=to_ras.out_file,
            airmask=amw.air_mask,
            hatmask=amw.hat_mask,
            artmask=amw.art_mask,
            rotmask=amw.rot_mask,
            segmentation=bts.out_segm,
            pvms=bts.out_pvms,
            headmask=hmsk.out_file,
            name="iqmswf",
        )
    )
    # Reports
    anat_report_wf = workflow.add(
        init_anat_report_wf(
            wf_species=wf_species,
            exec_verbose_reports=exec_verbose_reports,
            exec_work_dir=exec_work_dir,
            in_ras=to_ras.out_file,
            headmask=hmsk.out_file,
            airmask=amw.air_mask,
            artmask=amw.art_mask,
            segmentation=bts.out_segm,
            name="anat_report_wf",
        )
    )
    # Connect all nodes
    # fmt: off

    hmsk.inputs.in_file = skull_stripping.out_corrected
    hmsk.inputs.brainmask = skull_stripping.out_mask
    bts.inputs.brainmask = skull_stripping.out_mask
    norm.inputs.moving_image = skull_stripping.out_corrected
    norm.inputs.moving_mask = skull_stripping.out_mask
    hmsk.inputs.in_tpms = norm.out_tpms

    iqmswf.inputs.inu_corrected = skull_stripping.out_corrected
    iqmswf.inputs.in_inu = skull_stripping.bias_image
    iqmswf.inputs.brainmask = skull_stripping.out_mask

    anat_report_wf.inputs.brainmask = skull_stripping.out_mask

    # fmt: on
    # Upload metrics
    if not exec_no_sub:
        from pydra.tasks.mriqc.interfaces.webapi import UploadIQMs

        pass
        # fmt: off
        pass
        pass
        # fmt: on
    outputs_["norm_report"] = norm.report
    outputs_["iqmswf_noise_report"] = iqmswf.noise_report
    outputs_["anat_report_wf_segm_report"] = anat_report_wf.segm_report
    outputs_["anat_report_wf_bmask_report"] = anat_report_wf.bmask_report
    outputs_["anat_report_wf_artmask_report"] = anat_report_wf.artmask_report
    outputs_["anat_report_wf_airmask_report"] = anat_report_wf.airmask_report
    outputs_["anat_report_wf_bg_report"] = anat_report_wf.bg_report
    outputs_["anat_report_wf_headmask_report"] = anat_report_wf.headmask_report
    outputs_["anat_report_wf_zoom_report"] = anat_report_wf.zoom_report

    return tuple(outputs_)


@workflow.define(outputs=["hat_mask", "air_mask", "art_mask", "rot_mask"])
def airmsk_wf(
    head_mask: ty.Any = attrs.NOTHING,
    in_file: ty.Any = attrs.NOTHING,
    ind2std_xfm: ty.Any = attrs.NOTHING,
    name="AirMaskWorkflow",
) -> tuple[ty.Any, ty.Any, ty.Any, ty.Any]:
    """
    Calculate air, artifacts and "hat" masks to evaluate noise in the background.

    This workflow mostly addresses the implementation of Step 1 in [Mortamet2009]_.
    This work proposes to look at the signal distribution in the background, where
    no signals are expected, to evaluate the spread of the noise.
    It is in the background where [Mortamet2009]_ proposed to also look at the presence
    of ghosts and artifacts, where they are very easy to isolate.

    However, [Mortamet2009]_ proposes not to look at the background around the face
    because of the likely signal leakage through the phase-encoding axis sourcing from
    eyeballs (and their motion).
    To avoid that, [Mortamet2009]_ proposed atlas-based identification of two landmarks
    (nasion and cerebellar projection on to the occipital bone).
    MRIQC, for simplicity, used a such a mask created in MNI152NLin2009cAsym space and
    projected it on to the individual.
    Such a solution is inadequate because it doesn't drop full in-plane slices as there
    will be a large rotation of the individual's tilt of the head with respect to the
    template.
    The new implementation (23.1.x series) follows [Mortamet2009]_ more closely,
    projecting the two landmarks from the template space and leveraging
    *NiTransforms* to do that.

    .. workflow::

        from mriqc.testing import mock_config
        from mriqc.workflows.anatomical.base import airmsk_wf
        with mock_config():
            wf = airmsk_wf()

    """
    outputs_ = {
        "hat_mask": attrs.NOTHING,
        "air_mask": attrs.NOTHING,
        "art_mask": attrs.NOTHING,
        "rot_mask": attrs.NOTHING,
    }

    rotmsk = workflow.add(RotationMask(in_file=in_file), name="rotmsk")
    qi1 = workflow.add(
        ArtifactMask(head_mask=head_mask, in_file=in_file, ind2std_xfm=ind2std_xfm),
        name="qi1",
    )
    # fmt: off
    outputs_['hat_mask'] = qi1.out_hat_msk
    outputs_['air_mask'] = qi1.out_air_msk
    outputs_['art_mask'] = qi1.out_art_msk
    outputs_['rot_mask'] = rotmsk.out_file
    # fmt: on

    return tuple(outputs_)


@workflow.define(outputs=["out_file", "out_denoised"])
def headmsk_wf(
    brainmask: ty.Any = attrs.NOTHING,
    in_file: ty.Any = attrs.NOTHING,
    in_tpms: ty.Any = attrs.NOTHING,
    name="HeadMaskWorkflow",
    omp_nthreads=1,
    wf_species="human",
) -> tuple[ty.Any, ty.Any]:
    """
    Computes a head mask as in [Mortamet2009]_.

    .. workflow::

        from mriqc.testing import mock_config
        from mriqc.workflows.anatomical.base import headmsk_wf
        with mock_config():
            wf = headmsk_wf()

    """
    from pydra.tasks.niworkflows.interfaces.nibabel import ApplyMask

    outputs_ = {
        "out_file": attrs.NOTHING,
        "out_denoised": attrs.NOTHING,
    }

    def _select_wm(inlist):
        return [f for f in inlist if "WM" in f][0]

    enhance = workflow.add(
        python.define(_enhance, outputs=["out_file"])(in_file=in_file, wm_tpm=in_tpms),
        name="enhance",
    )
    gradient = workflow.add(
        python.define(image_gradient, outputs=["out_file"])(
            brainmask=brainmask, in_file=enhance.out_file
        ),
        name="gradient",
    )
    thresh = workflow.add(
        python.define(gradient_threshold, outputs=["out_file"])(
            brainmask=brainmask, in_file=gradient.out_file
        ),
        name="thresh",
    )
    if wf_species != "human":
        gradient.inputs.inputs.sigma = 3.0
        thresh.inputs.inputs.aniso = True
        thresh.inputs.inputs.thresh = 4.0
    apply_mask = workflow.add(
        ApplyMask(in_file=enhance.out_file, in_mask=brainmask), name="apply_mask"
    )
    # fmt: off
    enhance.inputs.wm_tpm = in_tpms
    outputs_['out_file'] = thresh.out_file
    outputs_['out_denoised'] = apply_mask.out_file
    # fmt: on

    return tuple(outputs_)


@workflow.define(outputs=["out_segm", "out_pvms"])
def init_brain_tissue_segmentation(
    brainmask: ty.Any = attrs.NOTHING,
    in_file: ty.Any = attrs.NOTHING,
    name="brain_tissue_segmentation",
    nipype_omp_nthreads=12,
    std_tpms: ty.Any = attrs.NOTHING,
) -> tuple[ty.Any, ty.Any]:
    """
    Setup a workflow for brain tissue segmentation.

    .. workflow::

        from mriqc.workflows.anatomical.base import init_brain_tissue_segmentation
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_brain_tissue_segmentation()

    """
    from pydra.tasks.ants.auto import Atropos

    outputs_ = {
        "out_segm": attrs.NOTHING,
        "out_pvms": attrs.NOTHING,
    }

    def _format_tpm_names(in_files, fname_string=None):
        import glob
        from pathlib import Path
        import nibabel as nb

        out_path = Path.cwd().absolute()
        # copy files to cwd and rename iteratively
        for count, fname in enumerate(in_files):
            img = nb.load(fname)
            extension = "".join(Path(fname).suffixes)
            out_fname = f"priors_{1 + count:02}{extension}"
            nb.save(img, Path(out_path, out_fname))
        if fname_string is None:
            fname_string = f"priors_%02d{extension}"
        out_files = [
            str(prior)
            for prior in glob.glob(str(Path(out_path, f"priors*{extension}")))
        ]
        # return path with c-style format string for Atropos
        file_format = str(Path(out_path, fname_string))
        return file_format, out_files

    format_tpm_names = workflow.add(
        python.define(_format_tpm_names, outputs=["file_format"])(
            execution={"keep_inputs": True, "remove_unnecessary_outputs": False},
            in_files=std_tpms,
        ),
        name="format_tpm_names",
    )
    segment = workflow.add(
        Atropos(
            initialization="PriorProbabilityImages",
            mrf_radius=[1, 1, 1],
            mrf_smoothing_factor=0.01,
            num_threads=nipype_omp_nthreads,
            number_of_tissue_classes=3,
            out_classified_image_name="segment.nii.gz",
            output_posteriors_name_template="segment_%02d.nii.gz",
            prior_weighting=0.1,
            save_posteriors=True,
            intensity_images=in_file,
            mask_image=brainmask,
        ),
        name="segment",
    )
    # fmt: off

    @python.define
    def format_tpm_names_file_format_to_segment_prior_image_callable(in_: ty.Any) -> ty.Any:
        return _pop(in_)

    format_tpm_names_file_format_to_segment_prior_image_callable = workflow.add(format_tpm_names_file_format_to_segment_prior_image_callable(in_=format_tpm_names.file_format), name="format_tpm_names_file_format_to_segment_prior_image_callable")

    segment.inputs.prior_image = format_tpm_names_file_format_to_segment_prior_image_callable.out
    outputs_['out_segm'] = segment.classified_image
    outputs_['out_pvms'] = segment.posteriors
    # fmt: on

    return tuple(outputs_)


@workflow.define(outputs=["report", "ind2std_xfm", "out_tpms"])
def spatial_normalization(
    exec_ants_float=False,
    exec_debug=False,
    modality: ty.Any = attrs.NOTHING,
    moving_image: ty.Any = attrs.NOTHING,
    moving_mask: ty.Any = attrs.NOTHING,
    name="SpatialNormalization",
    nipype_omp_nthreads=12,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
) -> tuple[ty.Any, ty.Any, ty.Any]:
    """Create a simplified workflow to perform fast spatial normalization."""
    from pydra.tasks.niworkflows.interfaces.reportlets.registration import (
        SpatialNormalizationRPT as RobustMNINormalization,
    )

    outputs_ = {
        "report": attrs.NOTHING,
        "ind2std_xfm": attrs.NOTHING,
        "out_tpms": attrs.NOTHING,
    }

    # Have the template id handy
    tpl_id = wf_template_id
    # Define workflow interface

    # Spatial normalization
    norm = workflow.add(
        RobustMNINormalization(
            flavor=["testing", "fast"][exec_debug],
            float=exec_ants_float,
            generate_report=True,
            num_threads=nipype_omp_nthreads,
            template=tpl_id,
            moving_image=moving_image,
            moving_mask=moving_mask,
            reference=modality,
        ),
        name="norm",
    )
    if wf_species.lower() == "human":
        norm.inputs.inputs.reference_mask = str(
            get_template(tpl_id, resolution=2, desc="brain", suffix="mask")
        )
    else:
        norm.inputs.inputs.reference_image = str(get_template(tpl_id, suffix="T2w"))
        norm.inputs.inputs.reference_mask = str(
            get_template(tpl_id, desc="brain", suffix="mask")[0]
        )
    # Project standard TPMs into T1w space
    tpms_std2t1w = workflow.add(
        ApplyTransforms(
            default_value=0,
            dimension=3,
            float=exec_ants_float,
            interpolation="Gaussian",
            reference_image=moving_image,
            transforms=norm.inverse_composite_transform,
        ),
        name="tpms_std2t1w",
    )
    tpms_std2t1w.inputs.inputs.input_image = [
        str(p)
        for p in get_template(
            wf_template_id,
            suffix="probseg",
            resolution=(1 if wf_species.lower() == "human" else None),
            label=["CSF", "GM", "WM"],
        )
    ]
    # fmt: off
    outputs_['ind2std_xfm'] = norm.composite_transform
    outputs_['report'] = norm.out_report
    outputs_['out_tpms'] = tpms_std2t1w.output_image
    # fmt: on

    return tuple(outputs_)


@workflow.define(outputs=["measures", "noise_report"])
def compute_iqms(
    airmask: ty.Any = attrs.NOTHING,
    artmask: ty.Any = attrs.NOTHING,
    brainmask: ty.Any = attrs.NOTHING,
    hatmask: ty.Any = attrs.NOTHING,
    headmask: ty.Any = attrs.NOTHING,
    in_inu: ty.Any = attrs.NOTHING,
    in_ras: ty.Any = attrs.NOTHING,
    inu_corrected: ty.Any = attrs.NOTHING,
    name="ComputeIQMs",
    pvms: ty.Any = attrs.NOTHING,
    rotmask: ty.Any = attrs.NOTHING,
    segmentation: ty.Any = attrs.NOTHING,
    std_tpms: ty.Any = attrs.NOTHING,
    wf_species="human",
) -> tuple[ty.Any, ty.Any]:
    """
    Setup the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.anatomical.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from pydra.tasks.mriqc.interfaces.anatomical import Harmonize

    outputs_ = {
        "measures": attrs.NOTHING,
        "noise_report": attrs.NOTHING,
    }

    from pydra.tasks.mriqc.workflows.utils import _tofloat

    # Add provenance

    # AFNI check smoothing
    fwhm_task = get_fwhmx()
    fwhm = workflow.add(fwhm_task, name="fwhm")
    # Harmonize
    homog = workflow.add(
        Harmonize(brain_mask=brainmask, in_file=inu_corrected, wm_mask=pvms),
        name="homog",
    )
    if wf_species.lower() != "human":
        homog.inputs.inputs.erodemsk = False
        homog.inputs.inputs.thresh = 0.8
    # Mortamet's QI2
    getqi2 = workflow.add(ComputeQI2(air_msk=hatmask, in_file=in_ras), name="getqi2")
    # Compute python-coded measures
    measures = workflow.add(
        StructuralQC(
            human=wf_species.lower() == "human",
            air_msk=airmask,
            artifact_msk=artmask,
            head_msk=headmask,
            in_bias=in_inu,
            in_file=in_ras,
            in_noinu=homog.out_file,
            in_pvms=pvms,
            in_segm=segmentation,
            mni_tpms=std_tpms,
            rot_msk=rotmask,
        ),
        name="measures",
    )

    def _getwm(inlist):
        return inlist[-1]

    # fmt: off


    homog.inputs.wm_mask = pvms

    @python.define
    def fwhm_fwhm_to_measures_in_fwhm_callable(in_: ty.Any) -> ty.Any:
        return _tofloat(in_)

    fwhm_fwhm_to_measures_in_fwhm_callable = workflow.add(
        fwhm_fwhm_to_measures_in_fwhm_callable(in_=fwhm.fwhm),
        name="fwhm_fwhm_to_measures_in_fwhm_callable"
    )

    measures.inputs.in_fwhm = fwhm_fwhm_to_measures_in_fwhm_callable.out
    outputs_['measures'] = measures.out_qc
    outputs_['noise_report'] = getqi2.out_file

    # fmt: on

    return tuple(outputs_)


def _enhance(in_file, wm_tpm, out_file=None):

    import nibabel as nb
    import numpy as np
    from pydra.tasks.mriqc.workflows.utils import generate_filename

    imnii = nb.load(in_file)
    data = imnii.get_fdata(dtype=np.float32)
    range_max = np.percentile(data[data > 0], 99.98)
    excess = data > range_max
    wm_prob = nb.load(wm_tpm).get_fdata()
    wm_prob[wm_prob < 0] = 0  # Ensure no negative values
    wm_prob[excess] = 0  # Ensure no outliers are considered
    # Calculate weighted mean and standard deviation
    wm_mu = np.average(data, weights=wm_prob)
    wm_sigma = np.sqrt(np.average((data - wm_mu) ** 2, weights=wm_prob))
    # Resample signal excess pixels
    data[excess] = np.random.normal(loc=wm_mu, scale=wm_sigma, size=excess.sum())
    out_file = out_file or str(generate_filename(in_file, suffix="enhanced").absolute())
    nb.Nifti1Image(data, imnii.affine, imnii.header).to_filename(out_file)
    return out_file


def _get_mod(in_file):

    from pathlib import Path

    in_file = Path(in_file)
    extension = "".join(in_file.suffixes)
    return in_file.name.replace(extension, "").split("_")[-1]


def _pop(inlist):

    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def gradient_threshold(in_file, brainmask, thresh=15.0, out_file=None, aniso=False):
    """Compute a threshold from the histogram of the magnitude gradient image"""
    import nibabel as nb
    import numpy as np
    from scipy import ndimage as sim
    from pydra.tasks.mriqc.workflows.utils import generate_filename

    if not aniso:
        struct = sim.iterate_structure(sim.generate_binary_structure(3, 2), 2)
    else:
        # Generate an anisotropic binary structure, taking into account slice thickness
        img = nb.load(in_file)
        zooms = img.header.get_zooms()
        dist = max(zooms)
        dim = img.header["dim"][0]
        x = np.ones((5) * np.ones(dim, dtype=np.int8))
        np.put(x, x.size // 2, 0)
        dist_matrix = np.round(sim.distance_transform_edt(x, sampling=zooms), 5)
        struct = dist_matrix <= dist
    imnii = nb.load(in_file)
    hdr = imnii.header.copy()
    hdr.set_data_dtype(np.uint8)
    data = imnii.get_fdata(dtype=np.float32)
    mask = np.zeros_like(data, dtype=np.uint8)
    mask[data > thresh] = 1
    mask = sim.binary_closing(mask, struct, iterations=2).astype(np.uint8)
    mask = sim.binary_erosion(mask, sim.generate_binary_structure(3, 2)).astype(
        np.uint8
    )
    segdata = np.asanyarray(nb.load(brainmask).dataobj) > 0
    segdata = sim.binary_dilation(segdata, struct, iterations=2, border_value=1).astype(
        np.uint8
    )
    mask[segdata] = 1
    # Remove small objects
    label_im, nb_labels = sim.label(mask)
    artmsk = np.zeros_like(mask)
    if nb_labels > 2:
        sizes = sim.sum(mask, label_im, list(range(nb_labels + 1)))
        ordered = sorted(zip(sizes, list(range(nb_labels + 1))), reverse=True)
        for _, label in ordered[2:]:
            mask[label_im == label] = 0
            artmsk[label_im == label] = 1
    mask = sim.binary_fill_holes(mask, struct).astype(
        np.uint8
    )  # pylint: disable=no-member
    out_file = out_file or str(generate_filename(in_file, suffix="gradmask").absolute())
    nb.Nifti1Image(mask, imnii.affine, hdr).to_filename(out_file)
    return out_file


def image_gradient(in_file, brainmask, sigma=4.0, out_file=None):
    """Computes the magnitude gradient of an image using numpy"""
    import nibabel as nb
    import numpy as np
    from scipy.ndimage import gaussian_gradient_magnitude as gradient
    from pydra.tasks.mriqc.workflows.utils import generate_filename

    imnii = nb.load(in_file)
    mask = np.bool_(nb.load(brainmask).dataobj)
    data = imnii.get_fdata(dtype=np.float32)
    datamax = np.percentile(data.reshape(-1), 99.5)
    data *= 100 / datamax
    data[mask] = 100
    zooms = np.array(imnii.header.get_zooms()[:3])
    sigma_xyz = 2 - zooms / min(zooms)
    grad = gradient(data, sigma * sigma_xyz)
    gradmax = np.percentile(grad.reshape(-1), 99.5)
    grad *= 100.0
    grad /= gradmax
    grad[mask] = 100
    out_file = out_file or str(generate_filename(in_file, suffix="grad").absolute())
    nb.Nifti1Image(grad, imnii.affine, imnii.header).to_filename(out_file)
    return out_file


def _binarize(in_file, threshold=0.5, out_file=None):

    import os.path as op
    import nibabel as nb
    import numpy as np

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath(f"{fname}_bin{ext}")
    nii = nb.load(in_file)
    data = nii.get_fdata() > threshold
    hdr = nii.header.copy()
    hdr.set_data_dtype(np.uint8)
    nb.Nifti1Image(data.astype(np.uint8), nii.affine, hdr).to_filename(out_file)
    return out_file
