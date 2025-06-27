import attrs
import logging
from pydra.compose import workflow
import typing as ty


logger = logging.getLogger(__name__)


@workflow.define(outputs=["out_brain", "bias_image", "out_mask", "out_corrected"])
def synthstrip_wf(
    in_files: ty.Any = attrs.NOTHING, name="synthstrip_wf", omp_nthreads=None
) -> ["ty.Any", "ty.Any", "ty.Any", "ty.Any"]:
    """Create a brain-extraction workflow using SynthStrip."""
    from pydra.tasks.ants.auto import N4BiasFieldCorrection

    outputs_ = {
        "out_brain": attrs.NOTHING,
        "bias_image": attrs.NOTHING,
        "out_mask": attrs.NOTHING,
        "out_corrected": attrs.NOTHING,
    }

    from pydra.tasks.niworkflows.interfaces.nibabel import ApplyMask, IntensityClip
    from pydra.tasks.mriqc.interfaces.synthstrip import SynthStrip

    # truncate target intensity for N4 correction
    pre_clip = workflow.add(
        IntensityClip(p_max=99.9, p_min=10, in_file=in_files), name="pre_clip"
    )
    pre_n4 = workflow.add(
        N4BiasFieldCorrection(
            copy_header=True,
            dimension=3,
            num_threads=omp_nthreads,
            rescale_intensities=True,
            input_image=pre_clip.out_file,
        ),
        name="pre_n4",
    )
    post_n4 = workflow.add(
        N4BiasFieldCorrection(
            copy_header=True,
            dimension=3,
            n_iterations=[50] * 4,
            num_threads=omp_nthreads,
            bias_image=True,
            input_image=pre_clip.out_file,
        ),
        name="post_n4",
    )
    synthstrip = workflow.add(
        SynthStrip(num_threads=omp_nthreads, in_file=pre_n4.output_image),
        name="synthstrip",
    )
    final_masked = workflow.add(
        ApplyMask(in_file=post_n4.output_image, in_mask=synthstrip.out_mask),
        name="final_masked",
    )
    # fmt: off
    post_n4.inputs.weight_image = synthstrip.out_mask
    outputs_['out_brain'] = final_masked.out_file
    outputs_['bias_image'] = post_n4.bias_image
    outputs_['out_mask'] = synthstrip.out_mask
    outputs_['out_corrected'] = post_n4.output_image
    # fmt: on

    return tuple(outputs_)
